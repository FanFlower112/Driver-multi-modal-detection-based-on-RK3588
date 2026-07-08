"""rPPG 后处理工具函数。

从原 PC 端 server_physmamba.py 的处理思路拆分而来，去除了 FastAPI、WebSocket、
PyTorch 推理，只保留：
1. 短窗口 rPPG 波形 FFT 心率估计
2. 长窗口 pulse buffer 拼接
3. 长窗口 BPM 估计
4. 信号质量判断
5. 稳定心率更新

LocalRPPGWorker / 测试脚本都可以复用本文件。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np


FS = 30.0
WINDOW_SIZE = 128
WINDOW_SECONDS = (WINDOW_SIZE - 1) / FS

# 长窗口缓存配置。沿用 server_physmamba.py 的思想：
# 每次模型输出 128 点短波形，将其按时间戳追加到 pulse_buffer，
# 再从最近若干秒的连续波形中估计更稳定的 BPM。
PULSE_KEEP_SECONDS = 30.0
# TinyRPPG-Lite 单窗口只有 128 点，短窗口容易出现 55/110 这种谐波锁定。
# 长窗口略微拉长，可以显著降低心率在二倍频/半频之间跳变。
LONG_BPM_SECONDS = 18.0
LONG_BPM_MIN_SECONDS = 8.0


def compute_bpm_from_signal(
    signal: Sequence[float],
    fs: float = FS,
    low_bpm: float = 45.0,
    high_bpm: float = 160.0,
    n_fft: int = 4096,
) -> Tuple[Optional[float], float, dict]:
    """从一段 rPPG/BVP 波形中估计 BPM。

    处理逻辑来自原 server_physmamba.py：
    - 去均值、二次多项式去趋势
    - 标准化
    - Hanning 加窗 + FFT
    - 在 45~160 BPM 内找峰值
    - 对高频二倍频做基频修正
    - 输出 confidence 和诊断信息
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
        "normal_prior_corrected": False,
        "normal_prior_bpm": None,
        "normal_prior_ratio": 0.0,
    }

    if x.size < 32:
        return None, 0.0, diag

    x = x - np.mean(x)
    signal_std = float(np.std(x))
    diag["signal_std"] = signal_std

    if signal_std < 1e-6:
        return None, 0.0, diag

    # 去趋势，削弱慢速漂移。
    t = np.arange(x.size, dtype=np.float32)
    try:
        coef = np.polyfit(t, x, deg=2)
        trend = np.polyval(coef, t)
        x = x - trend
    except Exception:
        x = x - np.mean(x)

    x = x - np.mean(x)
    x = x / (np.std(x) + 1e-6)

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

    def nearest_peak_power(target_bpm: float, tolerance: float = 8.0):
        if top_idx.size > 0:
            dist = np.abs(bpm_band[top_idx] - target_bpm)
            j = int(np.argmin(dist))
            if float(dist[j]) <= tolerance:
                idx = int(top_idx[j])
                return float(bpm_band[idx]), float(power_band[idx]), idx

        band = np.where(np.abs(bpm_band - target_bpm) <= tolerance)[0]
        if band.size > 0:
            idx = int(band[np.argmax(power_band[band])])
            return float(bpm_band[idx]), float(power_band[idx]), idx

        return None, None, None

    # 若最强峰落在较高 BPM 区间，检查其一半是否存在合理基频。
    # TinyRPPG-Lite 上不宜过度“除以 2”，否则很容易锁到 55 左右。
    # 因此这里比原 PC 端 PhysMamba 版本更保守：
    # - 只对 >=120 BPM 的高频峰做半频修正；
    # - 半频必须落在 60~95 BPM；
    # - 半频峰功率至少达到高频峰的 35%。
    if selected_bpm >= 120.0:
        half_bpm = selected_bpm / 2.0
        half_peak_bpm, half_peak_power, half_peak_idx = nearest_peak_power(
            half_bpm,
            tolerance=7.0,
        )
        if half_peak_bpm is not None and 60.0 <= half_peak_bpm <= 95.0:
            half_ratio = half_peak_power / (selected_power + 1e-12)
            if half_ratio >= 0.35:
                selected_bpm = float(half_peak_bpm)
                selected_idx = int(half_peak_idx)
                selected_power = float(half_peak_power)
                harmonic_corrected = True

    # 高频兜底：避免 130~160 BPM 的二倍频长期占优。
    # 同样调得更保守，避免把高频误改成 55~60。
    if selected_bpm >= 130.0:
        low_region = np.where((bpm_band >= 60.0) & (bpm_band <= 95.0))[0]
        if low_region.size > 0:
            low_idx = int(low_region[np.argmax(power_band[low_region])])
            low_bpm_value = float(bpm_band[low_idx])
            low_power = float(power_band[low_idx])
            low_ratio = low_power / (selected_power + 1e-12)
            if low_ratio >= 0.35:
                selected_bpm = low_bpm_value
                selected_idx = low_idx
                selected_power = low_power
                harmonic_corrected = True

    # 正常驾驶场景先验：静息/驾驶状态下多数人心率大多落在 62~100 BPM。
    # 当频谱最强峰落在 <60 或 >105 时，如果 Top Peaks 里存在一个 62~100 的候选峰，
    # 且功率不是特别弱，则优先采用该候选，减少 55/110 半频/二倍频锁定。
    normal_prior_corrected = False
    normal_prior_bpm = None
    normal_prior_ratio = 0.0
    normal_candidates = []
    for idx in top_idx:
        bpm_v = float(bpm_band[int(idx)])
        if 62.0 <= bpm_v <= 100.0:
            normal_candidates.append(int(idx))

    if normal_candidates and (selected_bpm < 60.0 or selected_bpm > 105.0):
        normal_idx = max(normal_candidates, key=lambda idx: float(power_band[idx]))
        normal_bpm = float(bpm_band[normal_idx])
        normal_power = float(power_band[normal_idx])
        normal_ratio = normal_power / (selected_power + 1e-12)
        if normal_ratio >= 0.18:
            selected_bpm = normal_bpm
            selected_idx = int(normal_idx)
            selected_power = normal_power
            normal_prior_corrected = True
            normal_prior_bpm = normal_bpm
            normal_prior_ratio = float(normal_ratio)

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
    diag["normal_prior_corrected"] = bool(normal_prior_corrected)
    diag["normal_prior_bpm"] = None if normal_prior_bpm is None else round(float(normal_prior_bpm), 1)
    diag["normal_prior_ratio"] = float(normal_prior_ratio)

    return float(selected_bpm), confidence, diag


def normalize_signal_for_frontend(signal: Sequence[float]) -> List[float]:
    """归一化到约 [-1, 1]，用于前端/Qt 波形显示。"""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return []
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    if m < 1e-8:
        return []
    x = x / m
    return [float(v) for v in x]


def resample_window(sample_buffer):
    """从带时间戳的帧缓存中，重采样出固定 WINDOW_SIZE 帧。

    sample_buffer 元素格式：
        (timestamp_seconds, frame)
    返回：
        selected_frames, effective_fps, target_times
    """
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


def append_pulse_buffer(
    pulse_buffer: Deque[Tuple[float, float]],
    target_times: Sequence[float],
    signal: Sequence[float],
    last_pulse_time: Optional[float],
) -> Optional[float]:
    """将当前 128 点 rPPG 窗口追加到长时间 pulse_buffer。"""
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


def estimate_long_bpm(pulse_buffer: Deque[Tuple[float, float]]):
    """使用长窗口 pulse_buffer 估计更稳定的 long_bpm。"""
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

    return (
        long_bpm,
        long_confidence,
        normalize_signal_for_frontend(uniform_x[-180:]),
        long_diag,
        duration,
    )


def cluster_values(values: Iterable[float], tolerance: float = 10.0):
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


def classify_quality(
    long_bpm,
    long_confidence: float,
    effective_fps,
    face_motion,
    crop_brightness,
    crop_contrast,
    long_duration: float,
):
    """
    TinyRPPG-Lite 版本的质量判断。

    原 server_physmamba.py 的置信度阈值更适合 PC 端 PhysMamba 输出。
    TinyRPPG-Lite 在 RKNN 上输出的频谱置信度通常更低，常见约 0.01~0.03。
    因此这里放宽阈值，避免 UI 一直显示“数据质量不足，不解读心率”。
    """
    reasons = []

    if long_bpm is None:
        if long_duration < LONG_BPM_MIN_SECONDS:
            return "warming", "正在积累 rPPG 长窗口数据"
        return "warming", "等待稳定心率"

    if long_duration < LONG_BPM_MIN_SECONDS:
        reasons.append("长窗口时间较短")

    # 之前这里用 12 FPS 做提示，很多 UVC 摄像头实际只有 8~12 FPS，
    # 但 rPPG 仍然可以给出参考心率；因此只在更低帧率时提示。
    if effective_fps is not None and effective_fps < 8.0:
        reasons.append("帧率偏低")

    # 车内光照复杂，阈值不要过严。
    if crop_brightness is not None and (crop_brightness < 35.0 or crop_brightness > 220.0):
        reasons.append("亮度异常")

    if crop_contrast is not None and crop_contrast < 8.0:
        reasons.append("对比度较低")

    # Haar 人脸框存在抖动，运动阈值放宽。
    if face_motion is not None and face_motion > 12.0:
        reasons.append("人脸运动较大")

    # TinyRPPG-Lite 置信度整体偏低。
    if long_confidence < 0.006:
        reasons.append("频谱较弱")

    # 明显不可用的情况才判 poor。
    if effective_fps is not None and effective_fps < 5.0:
        return "poor", ", ".join(reasons) if reasons else "帧率过低"

    if face_motion is not None and face_motion > 20.0:
        return "poor", ", ".join(reasons) if reasons else "人脸运动过大"

    # 适配 TinyRPPG-Lite 的质量分级。
    if long_confidence >= 0.035 and len(reasons) <= 1:
        return "good", ", ".join(reasons) if reasons else "信号稳定"

    if long_confidence >= 0.010 and len(reasons) <= 3:
        return "medium", ", ".join(reasons) if reasons else "信号可用"

    # 只要能得到 long_bpm，就允许作为参考心率输出。
    return "medium", ", ".join(reasons) if reasons else "低置信度参考心率"


def update_stable_bpm(
    long_bpm,
    quality: str,
    stable_bpm,
    stable_history,
    long_history,
    candidate_state: dict,
):
    """
    由 long_bpm 更新最终稳定心率 stable_bpm。

    TinyRPPG-Lite 版本：
    - 不再因为 poor / medium 过度拒绝初始化；
    - 只要 long_bpm 连续出现，就允许给 UI 一个参考心率；
    - 大跳变仍然做缓慢跟踪，避免界面心率乱跳。
    """
    if long_bpm is None:
        return stable_bpm, candidate_state, "等待 long bpm"

    long_bpm = float(long_bpm)

    # 限制明显不合理范围。
    # TinyRPPG-Lite 目前容易在 55/110 附近形成半频/二倍频锁定。
    # 对驾驶员状态展示而言，先把稳定心率输出限制在 58~145 BPM，
    # 低于 58 的结果不用于更新 stable_bpm，避免界面长期锁死在 55。
    if long_bpm < 58.0 or long_bpm > 145.0:
        return stable_bpm, candidate_state, "long bpm 超出显示范围"

    long_history.append(long_bpm)

    center, count = cluster_values(long_history, tolerance=12.0)

    # 初始化阶段：放宽，不再强依赖 quality。
    if stable_bpm is None:
        if center is not None and count >= 2:
            stable_history.append(center)
            return float(center), candidate_state, f"初始化稳定心率 cluster {count}"

        stable_history.append(long_bpm)
        return long_bpm, candidate_state, "初始化参考心率"

    target = center if center is not None and count >= 2 else long_bpm
    delta = abs(target - stable_bpm)

    # 小幅变化：正常跟踪。
    if delta <= 8.0:
        updated = 0.75 * stable_bpm + 0.25 * target
        stable_history.append(updated)
        candidate_state["bpm"] = None
        candidate_state["count"] = 0
        return float(np.median(np.asarray(stable_history))), candidate_state, "正常跟踪"

    # 中等变化：慢速跟踪。
    if delta <= 20.0:
        updated = 0.88 * stable_bpm + 0.12 * target
        stable_history.append(updated)
        candidate_state["bpm"] = None
        candidate_state["count"] = 0
        return float(np.median(np.asarray(stable_history))), candidate_state, "慢速调整"

    # 大跳变：需要连续出现才接受。
    candidate_bpm = candidate_state.get("bpm")
    candidate_count = int(candidate_state.get("count", 0))

    if candidate_bpm is None or abs(target - candidate_bpm) > 12.0:
        candidate_state["bpm"] = target
        candidate_state["count"] = 1
        return stable_bpm, candidate_state, "保持，等待跳变确认 1"

    candidate_state["bpm"] = 0.70 * candidate_bpm + 0.30 * target
    candidate_state["count"] = candidate_count + 1

    if candidate_state["count"] >= 3:
        updated = 0.90 * stable_bpm + 0.10 * candidate_state["bpm"]
        stable_history.append(updated)
        candidate_state["bpm"] = None
        candidate_state["count"] = 0
        return float(np.median(np.asarray(stable_history))), candidate_state, "接受持续变化"

    return stable_bpm, candidate_state, f"保持，等待跳变确认 {candidate_state['count']}"

