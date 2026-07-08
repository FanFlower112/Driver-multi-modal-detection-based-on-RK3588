"""各监测模块之间共享的轻量状态容器。"""

import threading
import time


class FusionState:
    # 眼动模块
    latest_eye_score = 0.0

    # YOLO 行为检测模块
    detected_behaviors = []
    shared_yolo_frame = None

    # rPPG 模块（当前仅预留接口，后续由通信线程写入）
    latest_rppg_bpm = None
    latest_rppg_long_bpm = None
    latest_rppg_instant_bpm = None
    latest_rppg_confidence = 0.0
    latest_rppg_quality = "waiting"
    latest_rppg_quality_reason = "等待上位机数据"
    latest_rppg_status = "未连接"
    latest_rppg_tracker_status = "waiting"
    latest_rppg_signal = []
    latest_rppg_sample_rate = 30.0
    latest_rppg_fps = None
    latest_rppg_pulse_seconds = 0.0
    latest_rppg_source_timestamp = None
    latest_rppg_received_at = None
    rppg_connected = False

    # RK3588 本地 rPPG 独立疲劳评分
    latest_rppg_fatigue_score = None
    latest_rppg_calibrated = False
    latest_rppg_baseline_progress = 0.0
    latest_rppg_hr_base = None
    latest_rppg_rmssd_base = None
    latest_rppg_rmssd_current = None
    latest_rppg_hr_drop_norm = 0.0
    latest_rppg_rmssd_rise_norm = 0.0
    latest_rppg_ppi_count = 0
    latest_rppg_bvp_seconds = 0.0
    latest_rppg_fatigue_status = "等待有效生理数据"

    _rppg_lock = threading.Lock()

    @staticmethod
    def _as_float(value, default=None):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def update_rppg(cls, payload):
        """写入一帧 rPPG 结果。

        后续 WebSocket/UDP/TCP 接收代码只需把解析后的字典传给本方法：
            FusionState.update_rppg(payload)

        字段与电脑端 server_physmamba.py 当前输出兼容。
        """
        if not isinstance(payload, dict):
            raise TypeError("rPPG payload 必须是 dict")

        signal = payload.get("signal", [])
        if not isinstance(signal, (list, tuple)):
            signal = []

        clean_signal = []
        for value in signal:
            number = cls._as_float(value)
            if number is not None:
                clean_signal.append(number)

        with cls._rppg_lock:
            cls.latest_rppg_bpm = cls._as_float(payload.get("bpm"))
            cls.latest_rppg_long_bpm = cls._as_float(payload.get("long_bpm"))
            cls.latest_rppg_instant_bpm = cls._as_float(payload.get("instant_bpm"))
            cls.latest_rppg_confidence = max(
                0.0,
                min(1.0, cls._as_float(payload.get("confidence"), 0.0)),
            )
            cls.latest_rppg_quality = str(payload.get("quality") or "waiting").lower()
            cls.latest_rppg_quality_reason = str(
                payload.get("quality_reason") or "暂无质量说明"
            )
            cls.latest_rppg_status = str(payload.get("status") or "waiting")
            cls.latest_rppg_tracker_status = str(
                payload.get("tracker_status") or "waiting"
            )
            cls.latest_rppg_signal = clean_signal
            cls.latest_rppg_sample_rate = cls._as_float(
                payload.get("sample_rate"), 30.0
            )
            cls.latest_rppg_fps = cls._as_float(payload.get("fps"))
            cls.latest_rppg_pulse_seconds = cls._as_float(
                payload.get("pulse_seconds"), 0.0
            )
            cls.latest_rppg_source_timestamp = payload.get("timestamp")
            cls.latest_rppg_fatigue_score = cls._as_float(
                payload.get("rppg_fatigue_score")
            )
            cls.latest_rppg_calibrated = bool(payload.get("rppg_calibrated", False))
            cls.latest_rppg_baseline_progress = max(
                0.0,
                min(1.0, cls._as_float(payload.get("rppg_baseline_progress"), 0.0)),
            )
            cls.latest_rppg_hr_base = cls._as_float(payload.get("rppg_hr_base"))
            cls.latest_rppg_rmssd_base = cls._as_float(payload.get("rppg_rmssd_base"))
            cls.latest_rppg_rmssd_current = cls._as_float(
                payload.get("rppg_rmssd_current")
            )
            cls.latest_rppg_hr_drop_norm = max(
                0.0,
                min(1.0, cls._as_float(payload.get("rppg_hr_drop_norm"), 0.0)),
            )
            cls.latest_rppg_rmssd_rise_norm = max(
                0.0,
                min(1.0, cls._as_float(payload.get("rppg_rmssd_rise_norm"), 0.0)),
            )
            cls.latest_rppg_ppi_count = int(
                cls._as_float(payload.get("rppg_ppi_count"), 0) or 0
            )
            cls.latest_rppg_bvp_seconds = max(
                0.0, cls._as_float(payload.get("rppg_bvp_seconds"), 0.0)
            )
            cls.latest_rppg_fatigue_status = str(
                payload.get("rppg_fatigue_status") or "等待有效生理数据"
            )
            cls.latest_rppg_received_at = time.time()
            cls.rppg_connected = True

    @classmethod
    def clear_rppg(cls, status="未连接"):
        """清空 rPPG 状态，可在网络断开时调用。"""
        with cls._rppg_lock:
            cls.latest_rppg_bpm = None
            cls.latest_rppg_long_bpm = None
            cls.latest_rppg_instant_bpm = None
            cls.latest_rppg_confidence = 0.0
            cls.latest_rppg_quality = "waiting"
            cls.latest_rppg_quality_reason = "等待上位机数据"
            cls.latest_rppg_status = status
            cls.latest_rppg_tracker_status = "waiting"
            cls.latest_rppg_signal = []
            cls.latest_rppg_fps = None
            cls.latest_rppg_pulse_seconds = 0.0
            cls.latest_rppg_source_timestamp = None
            cls.latest_rppg_fatigue_score = None
            cls.latest_rppg_calibrated = False
            cls.latest_rppg_baseline_progress = 0.0
            cls.latest_rppg_hr_base = None
            cls.latest_rppg_rmssd_base = None
            cls.latest_rppg_rmssd_current = None
            cls.latest_rppg_hr_drop_norm = 0.0
            cls.latest_rppg_rmssd_rise_norm = 0.0
            cls.latest_rppg_ppi_count = 0
            cls.latest_rppg_bvp_seconds = 0.0
            cls.latest_rppg_fatigue_status = "等待有效生理数据"
            cls.latest_rppg_received_at = None
            cls.rppg_connected = False

    @classmethod
    def get_rppg_snapshot(cls):
        """线程安全地读取当前 rPPG 状态。"""
        with cls._rppg_lock:
            return {
                "bpm": cls.latest_rppg_bpm,
                "long_bpm": cls.latest_rppg_long_bpm,
                "instant_bpm": cls.latest_rppg_instant_bpm,
                "confidence": cls.latest_rppg_confidence,
                "quality": cls.latest_rppg_quality,
                "quality_reason": cls.latest_rppg_quality_reason,
                "status": cls.latest_rppg_status,
                "tracker_status": cls.latest_rppg_tracker_status,
                "signal": list(cls.latest_rppg_signal),
                "sample_rate": cls.latest_rppg_sample_rate,
                "fps": cls.latest_rppg_fps,
                "pulse_seconds": cls.latest_rppg_pulse_seconds,
                "source_timestamp": cls.latest_rppg_source_timestamp,
                "rppg_fatigue_score": cls.latest_rppg_fatigue_score,
                "rppg_calibrated": cls.latest_rppg_calibrated,
                "rppg_baseline_progress": cls.latest_rppg_baseline_progress,
                "rppg_hr_base": cls.latest_rppg_hr_base,
                "rppg_rmssd_base": cls.latest_rppg_rmssd_base,
                "rppg_rmssd_current": cls.latest_rppg_rmssd_current,
                "rppg_hr_drop_norm": cls.latest_rppg_hr_drop_norm,
                "rppg_rmssd_rise_norm": cls.latest_rppg_rmssd_rise_norm,
                "rppg_ppi_count": cls.latest_rppg_ppi_count,
                "rppg_bvp_seconds": cls.latest_rppg_bvp_seconds,
                "rppg_fatigue_status": cls.latest_rppg_fatigue_status,
                "received_at": cls.latest_rppg_received_at,
                "connected": cls.rppg_connected,
            }
