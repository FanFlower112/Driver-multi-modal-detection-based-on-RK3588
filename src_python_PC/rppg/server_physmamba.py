import asyncio
import base64
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural_methods.model.PhysMamba import PhysMamba

app = FastAPI()

# ==================== RK3588 rPPG 结果输出 ====================
# 浏览器继续使用 /ws 上传摄像头画面；RK3588 使用 /rppg_ws 接收结果。
RPPG_CLIENTS = set()
RPPG_BROADCAST_INTERVAL = 0.25  # 每秒最多发送 4 次，足够 UI 实时显示


async def broadcast_rppg(payload):
    """向所有已连接的 RK3588 客户端广播一帧 rPPG 结果。"""
    if not RPPG_CLIENTS:
        return

    clients = list(RPPG_CLIENTS)
    results = await asyncio.gather(
        *(client.send_json(payload) for client in clients),
        return_exceptions=True,
    )

    for client, result in zip(clients, results):
        if isinstance(result, Exception):
            RPPG_CLIENTS.discard(client)


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index_physmamba.html"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FS = 30.0
WINDOW_SIZE = 128
WINDOW_SECONDS = (WINDOW_SIZE - 1) / FS
INFER_INTERVAL = 0.8
CROP_SCALE = 1.10
BOX_SMOOTH_ALPHA = 0.22
MAX_MISSED_FACE = 10

PULSE_KEEP_SECONDS = 22.0
LONG_BPM_SECONDS = 12.0
LONG_BPM_MIN_SECONDS = 8.0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

model_paths = []
model_paths += list(ROOT.glob("weights/UBFC_Intro_5070_FULL_Epoch17.pth"))
model_paths += list(ROOT.glob("logs/**/PreTrainedModels/UBFC_Intro_5070_FULL_Epoch17.pth"))
model_paths = sorted(model_paths)

if not model_paths:
    raise FileNotFoundError("没有找到 UBFC_Intro_5070_FULL_Epoch17.pth")

MODEL_PATH = model_paths[0]

print("device:", DEVICE)
print("model path:", MODEL_PATH)

model = PhysMamba().to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

new_state = {}
for k, v in state.items():
    if k.startswith("module."):
        new_state[k[len("module."):]] = v
    else:
        new_state[k] = v

missing, unexpected = model.load_state_dict(new_state, strict=False)
print("missing keys:", len(missing))
print("unexpected keys:", len(unexpected))

model.eval()
torch.backends.cudnn.benchmark = True

print("PhysMamba model loaded")


def expand_box(x, y, w, h, frame_w, frame_h, scale=CROP_SCALE):
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * scale
    nh = h * scale

    x1 = int(max(0, cx - nw / 2.0))
    y1 = int(max(0, cy - nh / 2.0))
    x2 = int(min(frame_w, cx + nw / 2.0))
    y2 = int(min(frame_h, cy + nh / 2.0))

    return x1, y1, x2, y2


def smooth_box(raw_box, prev_box, alpha=BOX_SMOOTH_ALPHA):
    if prev_box is None:
        return raw_box

    out = {}

    for key in ("x", "y", "w", "h"):
        out[key] = int(round((1.0 - alpha) * prev_box[key] + alpha * raw_box[key]))

    out["frame_w"] = raw_box["frame_w"]
    out["frame_h"] = raw_box["frame_h"]

    out["x"] = max(0, min(out["x"], out["frame_w"] - 1))
    out["y"] = max(0, min(out["y"], out["frame_h"] - 1))
    out["w"] = max(1, min(out["w"], out["frame_w"] - out["x"]))
    out["h"] = max(1, min(out["h"], out["frame_h"] - out["y"]))

    return out


def diff_normalize(frames):
    data = np.asarray(frames, dtype=np.float32) / 255.0
    diff = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)

    std = np.std(diff)
    if std < 1e-6:
        return None

    diff = diff / std
    pad = np.zeros_like(diff[:1])
    diff = np.concatenate([diff, pad], axis=0)

    x = torch.from_numpy(diff).float()
    x = x.permute(3, 0, 1, 2).unsqueeze(0)

    return x


def compute_bpm_from_signal(signal, fs=FS, low_bpm=45.0, high_bpm=160.0, n_fft=4096):
    """
    B版长窗口优化：FFT 选峰 + 二倍频/谐波修正。

    主要解决：
    真实心率约 65~80 BPM 时，频谱中 130~160 BPM 的二倍频可能更强，
    导致 Long BPM 被误判为 140 左右。
    """

    x = np.asarray(signal, dtype=np.float32).reshape(-1)

    diag = {
        "top_peaks": [],
        "peak_ratio": 0.0,
        "noise_ratio": 0.0,
        "signal_std": 0.0,
        "harmonic_corrected": False,
        "selected_before_harmonic": None,
        "harmonic_conflict": 0.0,
    }

    if x.size < 32:
        return None, 0.0, diag

    x = x - np.mean(x)
    signal_std = float(np.std(x))
    diag["signal_std"] = signal_std

    if signal_std < 1e-6:
        return None, 0.0, diag

    # 去趋势：减少慢速漂移对频谱的干扰
    t = np.arange(x.size, dtype=np.float32)
    try:
        coef = np.polyfit(t, x, deg=2)
        trend = np.polyval(coef, t)
        x = x - trend
    except Exception:
        x = x - np.mean(x)

    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-6)

    # 加窗 + FFT
    win = np.hanning(x.size).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x * win, n=n_fft)) ** 2
    bpm_axis = freqs * 60.0

    mask = (bpm_axis >= low_bpm) & (bpm_axis <= high_bpm)
    if not np.any(mask):
        return None, 0.0, diag

    bpm_band = bpm_axis[mask]
    power_band = spectrum[mask].astype(np.float64)

    if power_band.size < 3 or float(np.max(power_band)) <= 1e-12:
        return None, 0.0, diag

    # 找局部峰
    local_peak_idx = []
    for i in range(1, power_band.size - 1):
        if power_band[i] >= power_band[i - 1] and power_band[i] >= power_band[i + 1]:
            local_peak_idx.append(i)

    if not local_peak_idx:
        local_peak_idx = [int(np.argmax(power_band))]

    local_peak_idx = np.asarray(local_peak_idx, dtype=np.int64)
    order = np.argsort(power_band[local_peak_idx])[::-1]
    peak_idx = local_peak_idx[order]

    top_n = min(8, peak_idx.size)
    top_idx = peak_idx[:top_n]
    top_bpms = bpm_band[top_idx]

    selected_idx = int(top_idx[0])
    selected_bpm = float(bpm_band[selected_idx])
    selected_power = float(power_band[selected_idx])
    selected_before_harmonic = selected_bpm
    harmonic_corrected = False

    def nearest_peak_power(target_bpm, tolerance=8.0):
        # 优先从 Top Peaks 中找
        if top_idx.size > 0:
            dist = np.abs(bpm_band[top_idx] - target_bpm)
            j = int(np.argmin(dist))
            if float(dist[j]) <= tolerance:
                idx = int(top_idx[j])
                return float(bpm_band[idx]), float(power_band[idx]), idx

        # 再从完整频带附近找
        band = np.where(np.abs(bpm_band - target_bpm) <= tolerance)[0]
        if band.size > 0:
            idx = int(band[np.argmax(power_band[band])])
            return float(bpm_band[idx]), float(power_band[idx]), idx

        return None, None, None

    # 关键 1：如果选到高频峰，检查它的一半是否存在合理基频
    if selected_bpm >= 115.0:
        half_bpm = selected_bpm / 2.0
        half_peak_bpm, half_peak_power, half_peak_idx = nearest_peak_power(half_bpm, tolerance=8.0)

        if half_peak_bpm is not None and 45.0 <= half_peak_bpm <= 95.0:
            half_ratio = half_peak_power / (selected_power + 1e-12)

            # 基频不需要比二倍频强，只要不是特别弱，就优先认为它是真实心率
            if half_ratio >= 0.18:
                selected_bpm = float(half_peak_bpm)
                selected_idx = int(half_peak_idx)
                selected_power = float(half_peak_power)
                harmonic_corrected = True

    # 关键 2：低频心率区间兜底，避免 130~160 长时间占优
    if selected_bpm >= 125.0:
        low_region = np.where((bpm_band >= 55.0) & (bpm_band <= 95.0))[0]
        if low_region.size > 0:
            low_idx = int(low_region[np.argmax(power_band[low_region])])
            low_bpm = float(bpm_band[low_idx])
            low_power = float(power_band[low_idx])
            low_ratio = low_power / (selected_power + 1e-12)

            if low_ratio >= 0.22:
                selected_bpm = low_bpm
                selected_idx = low_idx
                selected_power = low_power
                harmonic_corrected = True

    # 谐波冲突：如果选了 70，但 140 也很强，置信度不要给太满
    harmonic_conflict = 0.0
    if 45.0 <= selected_bpm <= 110.0:
        double_bpm = selected_bpm * 2.0
        double_peak_bpm, double_peak_power, _ = nearest_peak_power(double_bpm, tolerance=8.0)
        if double_peak_bpm is not None:
            harmonic_conflict = float(double_peak_power / (selected_power + 1e-12))

    sorted_power = np.sort(power_band)[::-1]
    first_power = float(sorted_power[0])
    second_power = float(sorted_power[1]) if sorted_power.size > 1 else 1e-12
    median_power = float(np.median(power_band)) + 1e-12

    peak_ratio = first_power / (second_power + 1e-12)
    noise_ratio = selected_power / median_power

    confidence = 0.0
    confidence += min(1.0, max(0.0, (peak_ratio - 1.0) / 0.8)) * 0.30
    confidence += min(1.0, max(0.0, (noise_ratio - 1.0) / 5.0)) * 0.35
    confidence += min(1.0, signal_std / 0.25) * 0.20

    if harmonic_corrected:
        confidence += 0.15

    if harmonic_conflict > 1.8:
        confidence *= 0.60
    elif harmonic_conflict > 1.2:
        confidence *= 0.75
    elif harmonic_conflict > 0.8:
        confidence *= 0.88

    confidence = float(np.clip(confidence, 0.0, 1.0))

    diag["top_peaks"] = [round(float(v), 1) for v in top_bpms[:5]]
    diag["peak_ratio"] = float(peak_ratio)
    diag["noise_ratio"] = float(noise_ratio)
    diag["signal_std"] = float(signal_std)
    diag["harmonic_corrected"] = bool(harmonic_corrected)
    diag["selected_before_harmonic"] = round(float(selected_before_harmonic), 1)
    diag["harmonic_conflict"] = float(harmonic_conflict)

    return float(selected_bpm), confidence, diag


def normalize_signal_for_frontend(signal):
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    m = np.max(np.abs(x))

    if m < 1e-8:
        return []

    x = x / m

    return [float(v) for v in x]


def resample_window(sample_buffer):
    if len(sample_buffer) < 2:
        return None, None, None

    times = np.asarray([item[0] for item in sample_buffer], dtype=np.float64)
    frames = [item[1] for item in sample_buffer]

    end_t = times[-1]
    start_t = end_t - WINDOW_SECONDS

    if times[0] > start_t:
        return None, None, None

    target_times = np.linspace(start_t, end_t, WINDOW_SIZE)
    indices = np.searchsorted(times, target_times)

    selected = []

    for i, target in zip(indices, target_times):
        if i <= 0:
            idx = 0
        elif i >= len(times):
            idx = len(times) - 1
        else:
            left = i - 1
            right = i
            if abs(times[left] - target) <= abs(times[right] - target):
                idx = left
            else:
                idx = right

        selected.append(frames[idx])

    recent_mask = times >= start_t
    recent_count = int(np.sum(recent_mask))
    effective_fps = recent_count / max(WINDOW_SECONDS, 1e-6)

    return selected, effective_fps, target_times


def run_model(frames):
    x = diff_normalize(frames)

    if x is None:
        return None, None, 0.0, {}, "signal too weak"

    x = x.to(DEVICE)

    with torch.no_grad():
        y = model(x)

    if not torch.is_tensor(y):
        return None, None, 0.0, {}, "bad model output"

    signal = y.detach().float().cpu().numpy().reshape(-1)
    instant_bpm, instant_confidence, instant_diag = compute_bpm_from_signal(
        signal,
        fs=FS,
        low_bpm=45.0,
        high_bpm=160.0,
        n_fft=2048,
    )

    if instant_bpm is None:
        return None, normalize_signal_for_frontend(signal), instant_confidence, instant_diag, "bad bpm"

    return instant_bpm, normalize_signal_for_frontend(signal), instant_confidence, instant_diag, "model ok"


def append_pulse_buffer(pulse_buffer, target_times, signal, last_pulse_time):
    if target_times is None or signal is None:
        return last_pulse_time

    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    times = np.asarray(target_times, dtype=np.float64).reshape(-1)

    n = min(len(sig), len(times))
    if n < 2:
        return last_pulse_time

    sig = sig[:n]
    times = times[:n]

    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std > 1e-8:
        sig = sig / std

    new_last = last_pulse_time

    for t, v in zip(times, sig):
        t = float(t)
        if last_pulse_time is None or t > last_pulse_time + 1e-4:
            pulse_buffer.append((t, float(v)))
            new_last = t

    if new_last is not None:
        cutoff = new_last - PULSE_KEEP_SECONDS
        while pulse_buffer and pulse_buffer[0][0] < cutoff:
            pulse_buffer.popleft()

    return new_last


def estimate_long_bpm(pulse_buffer):
    if len(pulse_buffer) < int(LONG_BPM_MIN_SECONDS * FS * 0.5):
        return None, 0.0, [], {}, 0.0

    times = np.asarray([item[0] for item in pulse_buffer], dtype=np.float64)
    values = np.asarray([item[1] for item in pulse_buffer], dtype=np.float64)

    end_t = times[-1]
    start_t = max(times[0], end_t - LONG_BPM_SECONDS)
    mask = times >= start_t

    t = times[mask]
    x = values[mask]

    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0

    if duration < LONG_BPM_MIN_SECONDS:
        return None, 0.0, [], {}, duration

    target_t = np.arange(t[0], t[-1], 1.0 / FS)

    if len(target_t) < int(LONG_BPM_MIN_SECONDS * FS):
        return None, 0.0, [], {}, duration

    uniform_x = np.interp(target_t, t, x)
    long_bpm, long_confidence, long_diag = compute_bpm_from_signal(
        uniform_x,
        fs=FS,
        low_bpm=45.0,
        high_bpm=160.0,
        n_fft=4096,
    )

    return long_bpm, long_confidence, normalize_signal_for_frontend(uniform_x[-180:]), long_diag, duration


def cluster_values(values, tolerance=10.0):
    arr = np.asarray(list(values), dtype=np.float64)

    if len(arr) < 3:
        return None, 0

    best_center = None
    best_count = 0

    for v in arr:
        near = arr[np.abs(arr - v) <= tolerance]
        count = len(near)

        if count > best_count:
            best_count = count
            best_center = float(np.median(near))

    return best_center, best_count


def classify_quality(long_bpm, long_confidence, effective_fps, face_motion, crop_brightness, crop_contrast, long_duration):
    reasons = []

    if long_bpm is None:
        reasons.append("waiting long rPPG")
    if long_duration < LONG_BPM_MIN_SECONDS:
        reasons.append("short pulse buffer")
    if effective_fps is not None and effective_fps < 18.0:
        reasons.append("low fps")
    if crop_brightness is not None and (crop_brightness < 55.0 or crop_brightness > 190.0):
        reasons.append("bad brightness")
    if crop_contrast is not None and crop_contrast < 15.0:
        reasons.append("low contrast")
    if face_motion is not None and face_motion > 4.0:
        reasons.append("motion")
    if long_confidence < 0.05:
        reasons.append("weak spectrum")

    if long_bpm is None:
        return "warming", ", ".join(reasons)

    if face_motion is not None and face_motion > 7.0:
        return "poor", ", ".join(reasons)

    if effective_fps is not None and effective_fps < 15.0:
        return "poor", ", ".join(reasons)

    if long_confidence >= 0.22 and not reasons:
        return "good", "clean"

    if long_confidence >= 0.08 and len(reasons) <= 2:
        return "medium", ", ".join(reasons) if reasons else "usable"

    return "poor", ", ".join(reasons) if reasons else "low quality"


def update_stable_bpm(long_bpm, quality, stable_bpm, stable_history, long_history, candidate_state):
    if long_bpm is None:
        return stable_bpm, candidate_state, "no long bpm"

    long_bpm = float(long_bpm)
    long_history.append(long_bpm)

    center, count = cluster_values(long_history, tolerance=10.0)

    if stable_bpm is None:
        if count >= 3:
            stable_history.append(center)
            return center, candidate_state, f"init by long cluster {count}"

        if quality in ("good", "medium"):
            stable_history.append(long_bpm)
            return long_bpm, candidate_state, "init by long bpm"

        return None, candidate_state, f"waiting long cluster {count}/3"

    target = center if center is not None and count >= 3 else long_bpm
    delta = abs(target - stable_bpm)

    if quality == "poor":
        if center is not None and count >= 5 and delta <= 25.0:
            updated = 0.92 * stable_bpm + 0.08 * target
            stable_history.append(updated)
            return float(np.median(np.asarray(stable_history))), candidate_state, "poor but persistent slow update"

        return stable_bpm, candidate_state, "hold poor quality"

    if delta <= 8.0:
        updated = 0.78 * stable_bpm + 0.22 * target
        stable_history.append(updated)
        candidate_state["bpm"] = None
        candidate_state["count"] = 0
        return float(np.median(np.asarray(stable_history))), candidate_state, "track long bpm"

    if delta <= 20.0:
        updated = 0.88 * stable_bpm + 0.12 * target
        stable_history.append(updated)
        candidate_state["bpm"] = None
        candidate_state["count"] = 0
        return float(np.median(np.asarray(stable_history))), candidate_state, "slow long adjust"

    candidate_bpm = candidate_state.get("bpm")
    candidate_count = int(candidate_state.get("count", 0))

    if candidate_bpm is None or abs(target - candidate_bpm) > 10.0:
        candidate_state["bpm"] = target
        candidate_state["count"] = 1
        return stable_bpm, candidate_state, "hold long jump candidate 1"

    candidate_state["bpm"] = 0.70 * candidate_bpm + 0.30 * target
    candidate_state["count"] = candidate_count + 1

    if candidate_state["count"] >= 4:
        updated = 0.90 * stable_bpm + 0.10 * candidate_state["bpm"]
        stable_history.append(updated)
        candidate_state["bpm"] = None
        candidate_state["count"] = 0
        return float(np.median(np.asarray(stable_history))), candidate_state, "accept persistent long shift"

    return stable_bpm, candidate_state, f"hold long jump candidate {candidate_state['count']}"


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.websocket("/rppg_ws")
async def rppg_output_endpoint(websocket: WebSocket):
    """RK3588 专用接口：只发送 rPPG 结果，不接收摄像头帧。"""
    await websocket.accept()
    RPPG_CLIENTS.add(websocket)
    print(f"RK3588 rPPG client connected: {websocket.client}")

    try:
        # 等待客户端断开。结果由 broadcast_rppg() 主动发送。
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print("RK3588 rPPG connection error:", exc)
    finally:
        RPPG_CLIENTS.discard(websocket)
        print(f"RK3588 rPPG client disconnected: {websocket.client}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    sample_buffer = deque(maxlen=900)
    pulse_buffer = deque(maxlen=2000)
    stable_history = deque(maxlen=7)
    long_history = deque(maxlen=9)
    candidate_state = {"bpm": None, "count": 0}

    stable_bpm = None
    instant_bpm = None
    instant_confidence = 0.0
    long_bpm = None
    long_confidence = 0.0
    last_signal = []
    long_signal = []
    last_infer = 0.0
    last_pulse_time = None
    status = "waiting"
    latest_face = None
    smoothed_box = None
    missed_faces = 0
    effective_fps = None
    tracker_status = "waiting"
    quality = "warming"
    quality_reason = "waiting"
    diagnostics = {}
    prev_box_for_motion = None
    crop_brightness = None
    crop_contrast = None
    face_motion = None
    long_duration = 0.0
    last_rppg_broadcast_time = 0.0

    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)

            browser_ts = payload.get("ts")
            frame_ts = time.time() if browser_ts is None else float(browser_ts)

            data_url = payload.get("image", "")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]

            img_bytes = base64.b64decode(data_url)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({
                    "status": "bad frame",
                    "tracker_status": tracker_status,
                    "quality": quality,
                    "quality_reason": quality_reason,
                    "bpm": stable_bpm,
                    "long_bpm": long_bpm,
                    "instant_bpm": instant_bpm,
                    "confidence": long_confidence,
                    "instant_confidence": instant_confidence,
                    "face": latest_face,
                    "frames": len(sample_buffer),
                    "pulse_seconds": long_duration,
                    "signal": long_signal or last_signal,
                    "fps": effective_fps,
                    "diagnostics": diagnostics,
                })
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(35, 35),
            )

            latest_face = None

            if len(faces) == 0:
                status = "no face"
                missed_faces += 1
                if missed_faces > MAX_MISSED_FACE:
                    smoothed_box = None
            else:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, fw, fh = faces[0]

                rx1, ry1, rx2, ry2 = expand_box(x, y, fw, fh, w, h)

                raw_box = {
                    "x": int(rx1),
                    "y": int(ry1),
                    "w": int(rx2 - rx1),
                    "h": int(ry2 - ry1),
                    "frame_w": int(w),
                    "frame_h": int(h),
                }

                smoothed_box = smooth_box(raw_box, smoothed_box)
                latest_face = smoothed_box
                missed_faces = 0

                x1 = latest_face["x"]
                y1 = latest_face["y"]
                x2 = x1 + latest_face["w"]
                y2 = y1 + latest_face["h"]

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    status = "bad crop"
                else:
                    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_brightness = float(np.mean(gray_crop))
                    crop_contrast = float(np.std(gray_crop))

                    if prev_box_for_motion is None:
                        face_motion = 0.0
                    else:
                        cx0 = prev_box_for_motion["x"] + prev_box_for_motion["w"] / 2.0
                        cy0 = prev_box_for_motion["y"] + prev_box_for_motion["h"] / 2.0
                        cx1 = latest_face["x"] + latest_face["w"] / 2.0
                        cy1 = latest_face["y"] + latest_face["h"] / 2.0
                        face_motion = float(((cx1 - cx0) ** 2 + (cy1 - cx0 + cx0 - cy0) ** 2) ** 0.5)

                    prev_box_for_motion = dict(latest_face)

                    sample_buffer.append((frame_ts, crop))

                    window_frames, fps, target_times = resample_window(sample_buffer)

                    if fps is not None:
                        effective_fps = fps

                    if window_frames is None:
                        status = f"warming up {len(sample_buffer)} frames"
                    else:
                        now = time.time()
                        if now - last_infer >= INFER_INTERVAL:
                            last_infer = now
                            instant_bpm, sig, instant_confidence, instant_diag, model_status = run_model(window_frames)
                            status = model_status

                            if sig is not None:
                                last_signal = sig

                            if sig is not None and len(sig) == WINDOW_SIZE:
                                last_pulse_time = append_pulse_buffer(
                                    pulse_buffer,
                                    target_times,
                                    sig,
                                    last_pulse_time,
                                )

                            long_bpm, long_confidence, long_signal, long_diag, long_duration = estimate_long_bpm(pulse_buffer)

                            quality, quality_reason = classify_quality(
                                long_bpm,
                                long_confidence,
                                effective_fps,
                                face_motion,
                                crop_brightness,
                                crop_contrast,
                                long_duration,
                            )

                            stable_bpm, candidate_state, tracker_status = update_stable_bpm(
                                long_bpm,
                                quality,
                                stable_bpm,
                                stable_history,
                                long_history,
                                candidate_state,
                            )

                            diagnostics = dict(long_diag)
                            diagnostics["instant_top_peaks"] = instant_diag.get("top_peaks", [])
                            diagnostics["instant_peak_ratio"] = instant_diag.get("peak_ratio", 0.0)
                            diagnostics["instant_noise_ratio"] = instant_diag.get("noise_ratio", 0.0)
                            diagnostics["crop_brightness"] = crop_brightness
                            diagnostics["crop_contrast"] = crop_contrast
                            diagnostics["face_motion"] = face_motion

            result_payload = {
                "type": "rppg",
                "version": 1,
                "timestamp": time.time(),
                "status": status,
                "tracker_status": tracker_status,
                "quality": quality,
                "quality_reason": quality_reason,

                # bpm 对应网页中的 Stable BPM，也是 RK3588 的最终心率输入。
                "bpm": stable_bpm,
                "long_bpm": long_bpm,
                "instant_bpm": instant_bpm,
                "confidence": long_confidence,
                "instant_confidence": instant_confidence,

                "face": latest_face,
                "frames": len(sample_buffer),
                "pulse_seconds": long_duration,
                "signal": long_signal or last_signal,
                "sample_rate": FS,
                "fps": effective_fps,
                "diagnostics": diagnostics,
            }

            # 原电脑网页仍然接收相同的推理结果。
            await websocket.send_json(result_payload)

            # 同时按较低频率发送给 RK3588，避免无线网络和 UI 线程负担过大。
            now_broadcast = time.time()
            if now_broadcast - last_rppg_broadcast_time >= RPPG_BROADCAST_INTERVAL:
                await broadcast_rppg(result_payload)
                last_rppg_broadcast_time = now_broadcast

            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        print("client disconnected")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
