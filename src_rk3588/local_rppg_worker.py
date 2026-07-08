"""RK3588 本地 rPPG 推理线程。

方案 A：本线程独占摄像头 /dev/video21，直接完成：
摄像头采集 -> 人脸 ROI -> 128 帧缓存 -> tinyrppg.rknn 推理 -> rPPG 后处理 -> Qt 信号输出。

输出 payload 与原 rppg_receiver/server_physmamba.py 兼容，可直接连接到：
    RPPGWidget.update_rppg_data(payload)
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    RKNNLite = None
    _RKNN_IMPORT_ERROR = exc
else:
    _RKNN_IMPORT_ERROR = None

from rppg_postprocess import (
    FS,
    WINDOW_SIZE,
    append_pulse_buffer,
    classify_quality,
    compute_bpm_from_signal,
    estimate_long_bpm,
    normalize_signal_for_frontend,
    resample_window,
    update_stable_bpm,
)


MODEL_INPUT_SIZE = 64
CAMERA_DEFAULT = "/dev/video21"
INFER_INTERVAL = 1.0
SAMPLE_BUFFER_SECONDS = 30.0


class LocalRPPGWorker(QThread):
    data_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str, bool)

    def __init__(
        self,
        model_path="tinyrppg.rknn",
        camera=CAMERA_DEFAULT,
        width=640,
        height=480,
        camera_fps=30.0,
        infer_interval=INFER_INTERVAL,
        normalize="zero_one",
        fallback_center=True,
        parent=None,
    ):
        super().__init__(parent)
        self.model_path = str(model_path)
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.camera_fps = float(camera_fps)
        self.infer_interval = float(infer_interval)
        self.normalize = normalize
        self.fallback_center = bool(fallback_center)
        self.running = True

        # 图像帧缓存：元素为 (timestamp, 64x64 RGB face_roi)
        max_samples = int(max(300, SAMPLE_BUFFER_SECONDS * self.camera_fps))
        self.sample_buffer = deque(maxlen=max_samples)

        # rPPG 长窗口缓存
        self.pulse_buffer = deque(maxlen=2000)
        self.stable_history = deque(maxlen=7)
        self.long_history = deque(maxlen=9)
        self.candidate_state = {"bpm": None, "count": 0}

        self.rknn = None
        self.cap = None
        self.detector = None

        self.last_pulse_time = None
        self.stable_bpm = None
        self.instant_bpm = None
        self.instant_confidence = 0.0
        self.long_bpm = None
        self.long_confidence = 0.0
        self.last_signal = []
        self.long_signal = []
        self.quality = "warming"
        self.quality_reason = "waiting"
        self.status = "waiting"
        self.tracker_status = "waiting"
        self.effective_fps = None
        self.long_duration = 0.0
        self.diagnostics = {}

        self.smoothed_box = None
        self.prev_box_for_motion = None
        self.missed_faces = 0
        self.latest_face = None
        self.crop_brightness = None
        self.crop_contrast = None
        self.face_motion = None

    # ------------------------- 初始化 -------------------------

    def _load_face_detector(self):
        candidate_paths = []
        if hasattr(cv2, "data"):
            candidate_paths.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        candidate_paths += [
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
            Path("haarcascade_frontalface_default.xml"),
        ]

        for path in candidate_paths:
            if path.exists():
                detector = cv2.CascadeClassifier(str(path))
                if not detector.empty():
                    return detector, str(path)

        raise RuntimeError("未找到 Haar 人脸检测器 haarcascade_frontalface_default.xml")

    def _init_rknn(self):
        if RKNNLite is None:
            raise ImportError(
                "未找到 rknnlite.api.RKNNLite，请在 RK3588 上安装 rknn-toolkit-lite2"
            ) from _RKNN_IMPORT_ERROR

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 rPPG RKNN 模型: {model_path}")

        rknn = RKNNLite()
        ret = rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn 失败: {ret}")

        try:
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        except Exception:
            ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime 失败: {ret}")

        return rknn

    def _open_camera(self):
        camera_arg = self.camera
        if isinstance(camera_arg, str) and camera_arg.isdigit():
            camera_arg = int(camera_arg)

        # 指定 V4L2 后端，并优先请求 MJPG，很多 USB 摄像头在 YUYV 下只能跑 5~10 FPS，
        # 这会导致界面提示“帧率偏低”，也会影响 rPPG 稳定性。
        cap = cv2.VideoCapture(camera_arg, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_arg)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 rPPG 摄像头: {self.camera}")

        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[LocalRPPGWorker] camera opened: {self.camera}, size={real_w}x{real_h}, fps={real_fps}")
        return cap

    # ------------------------- 图像处理 -------------------------

    @staticmethod
    def _expand_box(x, y, w, h, frame_w, frame_h, scale=1.35):
        cx = x + w / 2.0
        cy = y + h / 2.0
        nw = w * scale
        nh = h * scale
        x1 = int(max(0, cx - nw / 2.0))
        y1 = int(max(0, cy - nh / 2.0))
        x2 = int(min(frame_w, cx + nw / 2.0))
        y2 = int(min(frame_h, cy + nh / 2.0))
        return x1, y1, x2, y2

    @staticmethod
    def _smooth_box(raw_box, prev_box, alpha=0.22):
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

    @staticmethod
    def _center_crop_box(frame, ratio=0.55):
        h, w = frame.shape[:2]
        crop_w = int(w * ratio)
        crop_h = int(h * ratio)
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "frame_w": w, "frame_h": h}

    def _detect_face_box(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        used_fallback = False
        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, fw, fh = faces[0]
            x1, y1, x2, y2 = self._expand_box(x, y, fw, fh, w, h, scale=1.35)
            raw_box = {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "frame_w": int(w),
                "frame_h": int(h),
            }
            self.missed_faces = 0
        else:
            self.missed_faces += 1
            if not self.fallback_center:
                return None, True
            raw_box = self._center_crop_box(frame)
            used_fallback = True

        # fallback 时也平滑，避免 ROI 跳变；连续丢脸太久则重置平滑框。
        if self.missed_faces > 10 and not used_fallback:
            self.smoothed_box = None
        self.smoothed_box = self._smooth_box(raw_box, self.smoothed_box)
        return self.smoothed_box, used_fallback

    def _crop_face_rgb64(self, frame):
        box, used_fallback = self._detect_face_box(frame)
        if box is None:
            return None, None, True

        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["w"]
        y2 = y1 + box["h"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, used_fallback

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.crop_brightness = float(np.mean(gray_crop))
        self.crop_contrast = float(np.std(gray_crop))

        if self.prev_box_for_motion is None:
            self.face_motion = 0.0
        else:
            cx0 = self.prev_box_for_motion["x"] + self.prev_box_for_motion["w"] / 2.0
            cy0 = self.prev_box_for_motion["y"] + self.prev_box_for_motion["h"] / 2.0
            cx1 = box["x"] + box["w"] / 2.0
            cy1 = box["y"] + box["h"] / 2.0
            self.face_motion = float(((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5)
        self.prev_box_for_motion = dict(box)

        crop = cv2.resize(crop, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb, box, used_fallback

    def _preprocess_clip(self, frames):
        clip = np.asarray(frames, dtype=np.float32)
        # clip: T,H,W,C = 128,64,64,3
        if clip.shape != (WINDOW_SIZE, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
            raise ValueError(f"rPPG 输入 clip shape 错误: {clip.shape}")

        if self.normalize == "zero_one":
            clip = clip / 255.0
        elif self.normalize == "minus_one_one":
            clip = clip / 127.5 - 1.0
        elif self.normalize == "standard":
            clip = clip / 255.0
            clip = (clip - np.mean(clip)) / (np.std(clip) + 1e-6)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"未知 normalize: {self.normalize}")

        # T,H,W,C -> 1,C,T,H,W
        clip = clip.transpose(3, 0, 1, 2)[None, ...]
        return np.ascontiguousarray(clip.astype(np.float32))

    # ------------------------- 推理和 payload -------------------------

    def _make_payload(self, status=None):
        """
        构造与原 WebSocket rPPG payload 兼容的数据包。

        为了避免 UI 因 quality=poor/warming 一直不显示心率：
        - bpm 优先使用 stable_bpm；没有则回退 long_bpm；再没有则回退 instant_bpm。
        - 如果已经有 bpm，但质量仍是 poor/warming，则给 UI 显示为 medium，说明是“参考心率”。
        """
        display_bpm = self.stable_bpm
        if display_bpm is None:
            display_bpm = self.long_bpm
        if display_bpm is None:
            display_bpm = self.instant_bpm

        display_quality = self.quality
        display_reason = self.quality_reason

        if display_bpm is not None and self.quality in ("poor", "warming"):
            display_quality = "medium"
            display_reason = f"本地 rPPG 参考心率，原质量状态：{self.quality_reason}"

        return {
            "type": "rppg",
            "version": 2,
            "timestamp": time.time(),
            "status": status or self.status,
            "tracker_status": self.tracker_status,
            "quality": display_quality,
            "quality_reason": display_reason,
            "bpm": display_bpm,
            "long_bpm": self.long_bpm,
            "instant_bpm": self.instant_bpm,
            "confidence": self.long_confidence,
            "instant_confidence": self.instant_confidence,
            "face": self.latest_face,
            "frames": len(self.sample_buffer),
            "pulse_seconds": self.long_duration,
            "signal": self.long_signal or self.last_signal,
            "sample_rate": FS,
            "fps": self.effective_fps,
            "diagnostics": self.diagnostics,
        }

    def _run_one_inference(self, window_frames, target_times):
        model_input = self._preprocess_clip(window_frames)

        t0 = time.time()
        outputs = self.rknn.inference(inputs=[model_input])
        infer_ms = (time.time() - t0) * 1000.0

        rppg = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        self.instant_bpm, self.instant_confidence, instant_diag = compute_bpm_from_signal(
            rppg,
            fs=FS,
            low_bpm=45.0,
            high_bpm=160.0,
            n_fft=2048,
        )
        self.status = "model ok" if self.instant_bpm is not None else "bad bpm"

        sig = normalize_signal_for_frontend(rppg)
        if sig:
            self.last_signal = sig
            if len(sig) == WINDOW_SIZE:
                self.last_pulse_time = append_pulse_buffer(
                    self.pulse_buffer,
                    target_times,
                    sig,
                    self.last_pulse_time,
                )

        self.long_bpm, self.long_confidence, self.long_signal, long_diag, self.long_duration = estimate_long_bpm(
            self.pulse_buffer
        )

        self.quality, self.quality_reason = classify_quality(
            self.long_bpm,
            self.long_confidence,
            self.effective_fps,
            self.face_motion,
            self.crop_brightness,
            self.crop_contrast,
            self.long_duration,
        )

        self.stable_bpm, self.candidate_state, self.tracker_status = update_stable_bpm(
            self.long_bpm,
            self.quality,
            self.stable_bpm,
            self.stable_history,
            self.long_history,
            self.candidate_state,
        )

        self.diagnostics = dict(long_diag)
        self.diagnostics.update({
            "instant_top_peaks": instant_diag.get("top_peaks", []),
            "instant_peak_ratio": instant_diag.get("peak_ratio", 0.0),
            "instant_noise_ratio": instant_diag.get("noise_ratio", 0.0),
            "instant_signal_std": instant_diag.get("signal_std", 0.0),
            "instant_harmonic_corrected": instant_diag.get("harmonic_corrected", False),
            "instant_normal_prior_corrected": instant_diag.get("normal_prior_corrected", False),
            "instant_selected_before_harmonic": instant_diag.get("selected_before_harmonic", None),
            "instant_normal_prior_bpm": instant_diag.get("normal_prior_bpm", None),
            "crop_brightness": self.crop_brightness,
            "crop_contrast": self.crop_contrast,
            "face_motion": self.face_motion,
            "infer_ms": infer_ms,
            "source": "rk3588_local_rknn",
            "normalize": self.normalize,
        })

    # ------------------------- QThread 主循环 -------------------------

    def run(self):
        try:
            self.detector, detector_path = self._load_face_detector()
            self.rknn = self._init_rknn()
            self.cap = self._open_camera()
            self.status_changed.emit(f"本地 rPPG 已启动：{self.camera}", True)
            print(f"[LocalRPPGWorker] Haar detector: {detector_path}")
            print(f"[LocalRPPGWorker] camera: {self.camera}")
            print(f"[LocalRPPGWorker] model: {self.model_path}")
        except Exception as exc:
            msg = f"本地 rPPG 启动失败：{exc}"
            print("[LocalRPPGWorker]", msg)
            self.status_changed.emit(msg, False)
            return

        last_infer = 0.0
        last_emit_waiting = 0.0

        while self.running and not self.isInterruptionRequested():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status = "camera read failed"
                self.quality = "waiting"
                self.quality_reason = "摄像头读取失败"
                now = time.time()
                if now - last_emit_waiting > 1.0:
                    self.data_ready.emit(self._make_payload())
                    last_emit_waiting = now
                self.msleep(30)
                continue

            now_mono = time.monotonic()
            crop_rgb, face_box, used_fallback = self._crop_face_rgb64(frame)

            if crop_rgb is None:
                self.status = "no face"
                self.quality = "waiting"
                self.quality_reason = "未检测到人脸"
                now = time.time()
                if now - last_emit_waiting > 1.0:
                    self.data_ready.emit(self._make_payload())
                    last_emit_waiting = now
                self.msleep(10)
                continue

            self.latest_face = face_box
            if used_fallback:
                self.status = "fallback center crop"

            self.sample_buffer.append((now_mono, crop_rgb))

            window_frames, fps, target_times = resample_window(self.sample_buffer)
            if fps is not None:
                self.effective_fps = float(fps)

            if window_frames is None:
                self.status = f"warming up {len(self.sample_buffer)} frames"
                self.quality = "warming"
                self.quality_reason = "正在缓存 128 帧人脸图像"
                now = time.time()
                if now - last_emit_waiting > 1.0:
                    self.data_ready.emit(self._make_payload())
                    last_emit_waiting = now
                self.msleep(1)
                continue

            now = time.time()
            if now - last_infer >= self.infer_interval:
                last_infer = now
                try:
                    self._run_one_inference(window_frames, target_times)
                except Exception as exc:
                    self.status = f"inference error: {exc}"
                    self.quality = "poor"
                    self.quality_reason = "本地 rPPG 推理异常"
                    print("[LocalRPPGWorker] inference error:", exc)

                self.data_ready.emit(self._make_payload())

            self.msleep(1)

        self._release()

    def _release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass
            self.rknn = None
        self.status_changed.emit("本地 rPPG 已停止", False)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.wait(3000)
        self._release()
