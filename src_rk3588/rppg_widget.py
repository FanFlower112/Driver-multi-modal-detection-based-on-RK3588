"""rPPG 生理状态页面。

当前版本只实现 UI 和数据接口，不主动建立网络连接。
后续接收线程拿到电脑端 JSON 后，调用：
    rppg_widget.update_rppg_data(payload)
或者直接调用：
    FusionState.update_rppg(payload)
即可刷新本页面与综合页面。
"""

import math
import time
from collections import deque

from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fusion_state import FusionState
from ui_theme import ACCENT, BORDER, DANGER, MUTED, PLOT, SUCCESS, TEXT, WARNING


class PulseWaveWidget(QWidget):
    """轻量波形控件，避免为单条曲线引入额外绘图库开销。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal = []
        self._sample_rate = 30.0
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_signal(self, signal, sample_rate=30.0):
        clean = []
        for value in signal or []:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                clean.append(value)

        if clean:
            peak = max(abs(v) for v in clean)
            if peak > 1.2:
                clean = [v / peak for v in clean]

        self._signal = clean
        self._sample_rate = sample_rate if sample_rate and sample_rate > 0 else 30.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(PLOT))

        margin_left = 52
        margin_right = 18
        margin_top = 24
        margin_bottom = 38
        plot_w = max(1, self.width() - margin_left - margin_right)
        plot_h = max(1, self.height() - margin_top - margin_bottom)

        grid_pen = QPen(QColor(BORDER), 1)
        painter.setPen(grid_pen)
        for i in range(5):
            y = margin_top + i * plot_h / 4
            painter.drawLine(margin_left, int(y), margin_left + plot_w, int(y))
        for i in range(7):
            x = margin_left + i * plot_w / 6
            painter.drawLine(int(x), margin_top, int(x), margin_top + plot_h)

        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Sans Serif", 10))
        painter.drawText(8, margin_top + 5, "+1")
        painter.drawText(18, margin_top + plot_h // 2 + 5, "0")
        painter.drawText(10, margin_top + plot_h + 5, "-1")
        painter.drawText(
            margin_left,
            self.height() - 12,
            "时间（最近窗口）    ·    纵轴：归一化相对脉搏幅度",
        )

        if len(self._signal) < 2:
            painter.setPen(QColor(MUTED))
            painter.setFont(QFont("Sans Serif", 15))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "等待电脑端 rPPG 波形数据…",
            )
            return

        path = QPainterPath()
        count = len(self._signal)
        for index, value in enumerate(self._signal):
            value = max(-1.0, min(1.0, value))
            x = margin_left + index / max(1, count - 1) * plot_w
            y = margin_top + (1.0 - (value + 1.0) / 2.0) * plot_h
            point = QPointF(x, y)
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)

        painter.setPen(QPen(QColor(ACCENT), 2.2))
        painter.drawPath(path)

        duration = (count - 1) / self._sample_rate
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Sans Serif", 10))
        painter.drawText(
            margin_left + plot_w - 80,
            self.height() - 12,
            f"{duration:.1f} 秒",
        )


class MetricCard(QFrame):
    def __init__(self, title, value="--", detail="", parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")

        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("MutedLabel")
        self.detail_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)
        layout.addStretch(1)

    def set_value(self, value, detail=None, color=TEXT):
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color};")
        if detail is not None:
            self.detail_label.setText(detail)


class RPPGFatigueEstimator:
    """RK3588 本地 rPPG 独立疲劳评分器。

    使用 60 秒、25 Hz BVP 环形缓冲区提取 PPI 与 RMSSD；前 120 秒有效
    数据建立清醒基线；随后按 HR 相对下降率和 RMSSD 相对上升率计算
    0～100 分。只使用 Python 标准库，不依赖 SciPy 或额外模型。
    """

    TARGET_FS = 25.0
    WINDOW_SECONDS = 60.0
    BASELINE_SECONDS = 120.0

    HR_WEIGHT = 0.4
    RMSSD_WEIGHT = 0.6
    HR_DROP_MAX = 0.20
    RMSSD_RISE_MAX = 1.00

    MIN_CONFIDENCE = 0.05
    RESET_GAP_SECONDS = 5.0

    def __init__(self):
        self._max_samples = int(self.TARGET_FS * self.WINDOW_SECONDS)
        self.reset()

    def reset(self):
        self.bvp_buffer = deque(maxlen=self._max_samples)
        self.baseline_hr_values = []
        self.baseline_rmssd_values = []
        self.valid_baseline_seconds = 0.0
        self.hr_base = None
        self.rmssd_base = None
        self.calibrated = False
        self.last_score = None
        self.last_unique_signal_time = None
        self.last_signal_fingerprint = None
        self.last_rmssd = None
        self.last_ppi_count = 0
        self.last_status = "等待有效生理数据"

    @staticmethod
    def _as_float(value, default=None):
        try:
            if value is None:
                return default
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip(value, low=0.0, high=1.0):
        return max(low, min(float(value), high))

    @staticmethod
    def _mean(values):
        return sum(values) / len(values) if values else None

    @staticmethod
    def _median(values):
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @staticmethod
    def _sanitize_signal(signal):
        clean = []
        if not isinstance(signal, (list, tuple)):
            return clean
        for value in signal:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean.append(number)
        return clean

    def _resample(self, values, source_fs):
        if len(values) < 2:
            return values
        source_fs = self._as_float(source_fs, 30.0)
        if source_fs is None or source_fs <= 0:
            source_fs = 30.0
        if abs(source_fs - self.TARGET_FS) < 1e-6:
            return list(values)

        duration = (len(values) - 1) / source_fs
        target_count = max(2, int(round(duration * self.TARGET_FS)) + 1)
        out = []
        for index in range(target_count):
            source_pos = (index / self.TARGET_FS) * source_fs
            left = min(int(math.floor(source_pos)), len(values) - 1)
            right = min(left + 1, len(values) - 1)
            frac = source_pos - left
            out.append(values[left] * (1.0 - frac) + values[right] * frac)
        return out

    @staticmethod
    def _fingerprint(values):
        if not values:
            return None
        tail = values[-min(48, len(values)):]
        return (len(values), tuple(round(value, 5) for value in tail))

    def _append_signal(self, signal, source_fs, now):
        clean = self._sanitize_signal(signal)
        if len(clean) < 2:
            return 0, 0.0

        resampled = self._resample(clean, source_fs)
        fingerprint = self._fingerprint(resampled)
        if fingerprint == self.last_signal_fingerprint:
            return 0, 0.0

        if (
            self.last_unique_signal_time is not None
            and now - self.last_unique_signal_time > self.RESET_GAP_SECONDS
        ):
            self.reset()

        if self.last_unique_signal_time is None:
            append_count = min(len(resampled), self._max_samples)
        else:
            elapsed = max(1.0 / self.TARGET_FS, now - self.last_unique_signal_time)
            append_count = int(round(min(elapsed, 2.0) * self.TARGET_FS))
            append_count = max(1, min(append_count, len(resampled)))

        self.bvp_buffer.extend(resampled[-append_count:])
        self.last_signal_fingerprint = fingerprint
        self.last_unique_signal_time = now
        return append_count, append_count / self.TARGET_FS

    @staticmethod
    def _moving_average(values, window=5):
        if len(values) < window or window <= 1:
            return list(values)
        half = window // 2
        prefix = [0.0]
        for value in values:
            prefix.append(prefix[-1] + value)
        out = []
        for index in range(len(values)):
            left = max(0, index - half)
            right = min(len(values), index + half + 1)
            out.append((prefix[right] - prefix[left]) / (right - left))
        return out

    def _detect_ppi_for_orientation(self, values, invert=False):
        if len(values) < int(self.TARGET_FS * 10):
            return [], None

        x = [-value for value in values] if invert else list(values)
        x = self._moving_average(x, window=5)
        mean_value = self._mean(x)
        variance = self._mean([(value - mean_value) ** 2 for value in x])
        std = math.sqrt(max(variance or 0.0, 0.0))
        if std < 1e-8:
            return [], None

        threshold = mean_value + 0.10 * std
        min_distance = max(1, int(self.TARGET_FS * 60.0 / 190.0))
        prominence_radius = max(2, int(0.20 * self.TARGET_FS))
        peaks = []

        for index in range(1, len(x) - 1):
            if not (
                x[index] > x[index - 1]
                and x[index] >= x[index + 1]
                and x[index] >= threshold
            ):
                continue

            left = x[max(0, index - prominence_radius):index]
            right = x[index + 1:min(len(x), index + prominence_radius + 1)]
            if not left or not right:
                continue
            prominence = x[index] - max(min(left), min(right))
            if prominence < 0.12 * std:
                continue

            if peaks and index - peaks[-1] < min_distance:
                if x[index] > x[peaks[-1]]:
                    peaks[-1] = index
            else:
                peaks.append(index)

        if len(peaks) < 4:
            return [], None

        ppi = []
        for left, right in zip(peaks, peaks[1:]):
            interval_ms = (right - left) * 1000.0 / self.TARGET_FS
            if 300.0 <= interval_ms <= 1500.0:
                ppi.append(interval_ms)
        if len(ppi) < 3:
            return [], None

        median_ppi = self._median(ppi)
        filtered = [
            value for value in ppi
            if 0.65 * median_ppi <= value <= 1.35 * median_ppi
        ]
        if len(filtered) < 3:
            return [], None

        return filtered, 60000.0 / self._median(filtered)

    def _extract_rmssd(self, current_bpm):
        values = list(self.bvp_buffer)
        if len(values) < int(self._max_samples * 0.85):
            return None, 0

        candidates = []
        for invert in (False, True):
            ppi, estimated_hr = self._detect_ppi_for_orientation(values, invert)
            if len(ppi) < 3 or estimated_hr is None:
                continue
            if current_bpm is not None and current_bpm > 0:
                cost = abs(estimated_hr - current_bpm)
            else:
                mean_ppi = self._mean(ppi)
                spread = math.sqrt(self._mean([(value - mean_ppi) ** 2 for value in ppi]))
                cost = spread / max(mean_ppi, 1.0) * 100.0
            cost -= min(len(ppi), 100) * 0.02
            candidates.append((cost, ppi))

        if not candidates:
            return None, 0

        ppi = min(candidates, key=lambda item: item[0])[1]
        diff_sq = [
            (ppi[index] - ppi[index - 1]) ** 2
            for index in range(1, len(ppi))
        ]
        if not diff_sq:
            return None, len(ppi)

        rmssd = math.sqrt(self._mean(diff_sq))
        if not math.isfinite(rmssd) or rmssd <= 0:
            return None, len(ppi)
        return rmssd, len(ppi)

    def _result(self, score=None, status=None, hr_norm=0.0, rmssd_norm=0.0):
        progress = self._clip(self.valid_baseline_seconds / self.BASELINE_SECONDS)
        return {
            "rppg_fatigue_score": score,
            "rppg_calibrated": self.calibrated,
            "rppg_baseline_progress": progress,
            "rppg_hr_base": self.hr_base,
            "rppg_rmssd_base": self.rmssd_base,
            "rppg_rmssd_current": self.last_rmssd,
            "rppg_hr_drop_norm": self._clip(hr_norm),
            "rppg_rmssd_rise_norm": self._clip(rmssd_norm),
            "rppg_ppi_count": self.last_ppi_count,
            "rppg_bvp_seconds": len(self.bvp_buffer) / self.TARGET_FS,
            "rppg_fatigue_status": status or self.last_status,
        }

    def update(self, payload):
        now = time.monotonic()
        bpm = self._as_float(payload.get("bpm"))
        quality = str(payload.get("quality") or "waiting").lower()
        confidence = self._as_float(payload.get("confidence"), 0.0) or 0.0
        source_fs = self._as_float(payload.get("sample_rate"), 30.0) or 30.0

        new_count, appended_seconds = self._append_signal(
            payload.get("signal", []), source_fs, now
        )
        valid_bpm = bpm is not None and 35.0 <= bpm <= 200.0
        quality_ok = quality in ("good", "medium") and confidence >= self.MIN_CONFIDENCE

        rmssd, ppi_count = self._extract_rmssd(bpm if valid_bpm else None)
        self.last_rmssd = rmssd
        self.last_ppi_count = ppi_count

        if not valid_bpm:
            self.last_status = "等待稳定心率数据"
            return self._result(self.last_score, self.last_status)
        if new_count <= 0:
            return self._result(self.last_score, self.last_status)
        if not quality_ok:
            self.last_status = (
                "当前信号质量不足，保持上一疲劳分"
                if self.calibrated and self.last_score is not None
                else "信号质量不足，基线建模暂停"
            )
            return self._result(self.last_score, self.last_status)

        if not self.calibrated:
            self.valid_baseline_seconds += appended_seconds
            self.baseline_hr_values.append(bpm)
            if rmssd is not None:
                self.baseline_rmssd_values.append(rmssd)

            progress = self._clip(self.valid_baseline_seconds / self.BASELINE_SECONDS)
            remaining = max(0, int(math.ceil(self.BASELINE_SECONDS - self.valid_baseline_seconds)))
            bvp_seconds = len(self.bvp_buffer) / self.TARGET_FS
            if bvp_seconds < self.WINDOW_SECONDS:
                self.last_status = (
                    f"正在积累60秒脉搏波：{bvp_seconds:.0f}/60秒；"
                    f"基线进度 {progress * 100:.0f}%"
                )
            else:
                self.last_status = (
                    f"清醒生理基线建模中：{progress * 100:.0f}%（约剩{remaining}秒）"
                )

            if (
                self.valid_baseline_seconds >= self.BASELINE_SECONDS
                and len(self.baseline_hr_values) >= 30
                and len(self.baseline_rmssd_values) >= 10
            ):
                self.hr_base = self._mean(self.baseline_hr_values)
                self.rmssd_base = self._mean(self.baseline_rmssd_values)
                if self.hr_base and self.rmssd_base and self.rmssd_base > 1e-6:
                    self.calibrated = True
                    self.last_score = 0.0
                    self.last_status = "清醒生理基线已锁定"
                else:
                    self.last_status = "基线数据无效，请改善光照并保持静止"

            return self._result(
                0.0 if self.baseline_hr_values else None,
                self.last_status,
            )

        if rmssd is None or self.hr_base is None or self.rmssd_base is None:
            self.last_status = "PPI数量不足，保持上一疲劳分"
            return self._result(self.last_score, self.last_status)

        x1 = max(0.0, (self.hr_base - bpm) / max(self.hr_base, 1e-6))
        x2 = max(0.0, (rmssd - self.rmssd_base) / max(self.rmssd_base, 1e-6))
        x1_norm = self._clip(x1 / self.HR_DROP_MAX)
        x2_norm = self._clip(x2 / self.RMSSD_RISE_MAX)

        weight_sum = self.HR_WEIGHT + self.RMSSD_WEIGHT
        score = (
            x1_norm * self.HR_WEIGHT + x2_norm * self.RMSSD_WEIGHT
        ) / weight_sum * 100.0
        self.last_score = self._clip(score, 0.0, 100.0)
        self.last_status = (
            f"HR下降特征 {x1_norm * 100:.0f}%；"
            f"RMSSD上升特征 {x2_norm * 100:.0f}%"
        )
        return self._result(self.last_score, self.last_status, x1_norm, x2_norm)


class RPPGWidget(QWidget):
    # 后续网络线程可以安全地调用 update_rppg_data()，信号会切回 Qt 主线程。
    external_data = pyqtSignal(dict)

    QUALITY_TEXT = {
        "good": "良好",
        "medium": "一般",
        "poor": "较差",
        "warming": "预热中",
        "waiting": "等待中",
    }

    QUALITY_COLOR = {
        "good": SUCCESS,
        "medium": WARNING,
        "poor": DANGER,
        "warming": MUTED,
        "waiting": MUTED,
    }

    def __init__(self):
        super().__init__()
        self.fatigue_estimator = RPPGFatigueEstimator()
        self.external_data.connect(self._accept_external_data)
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(250)
        self.refresh_ui()

    def _build_ui(self):
        title = QLabel("rPPG 生理状态监测")
        title.setObjectName("PageTitle")
        subtitle = QLabel("展示电脑端 PhysMamba 输出，并在 RK3588 本地计算独立 rPPG 疲劳度")
        subtitle.setObjectName("PageSubtitle")

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.connection_label = QLabel("●  等待接入")
        self.connection_label.setAlignment(Qt.AlignCenter)
        self.connection_label.setStyleSheet(
            f"color: {MUTED}; background: {PLOT}; border: 1px solid {BORDER}; "
            "border-radius: 12px; padding: 9px 16px; font-size: 14px; font-weight: 600;"
        )

        header = QHBoxLayout()
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(self.connection_label)

        wave_card = QFrame()
        wave_card.setObjectName("Card")
        wave_title = QLabel("类脉搏波形")
        wave_title.setObjectName("SectionTitle")
        wave_tip = QLabel("波形纵轴为归一化相对幅度，不代表血压、血氧或绝对血容量")
        wave_tip.setObjectName("MutedLabel")
        self.wave_widget = PulseWaveWidget()

        wave_layout = QVBoxLayout(wave_card)
        wave_layout.setContentsMargins(18, 16, 18, 18)
        wave_layout.setSpacing(10)
        wave_layout.addWidget(wave_title)
        wave_layout.addWidget(wave_tip)
        wave_layout.addWidget(self.wave_widget, 1)

        bpm_card = QFrame()
        bpm_card.setObjectName("Card")
        bpm_title = QLabel("稳定心率")
        bpm_title.setObjectName("SectionTitle")
        self.bpm_label = QLabel("--")
        self.bpm_label.setAlignment(Qt.AlignCenter)
        self.bpm_label.setStyleSheet(
            f"color: {TEXT}; font-size: 74px; font-weight: 800; background: transparent;"
        )
        bpm_unit = QLabel("BPM")
        bpm_unit.setAlignment(Qt.AlignCenter)
        bpm_unit.setStyleSheet(f"color: {MUTED}; font-size: 18px;")
        self.interpretation_label = QLabel("等待有效 rPPG 数据")
        self.interpretation_label.setAlignment(Qt.AlignCenter)
        self.interpretation_label.setWordWrap(True)
        self.interpretation_label.setStyleSheet(
            f"color: {MUTED}; background: {PLOT}; border-radius: 12px; "
            "padding: 12px; font-size: 16px;"
        )

        bpm_layout = QVBoxLayout(bpm_card)
        bpm_layout.setContentsMargins(20, 18, 20, 20)
        bpm_layout.addWidget(bpm_title)
        bpm_layout.addStretch(1)
        bpm_layout.addWidget(self.bpm_label)
        bpm_layout.addWidget(bpm_unit)
        bpm_layout.addSpacing(18)
        bpm_layout.addWidget(self.interpretation_label)
        bpm_layout.addStretch(1)

        fatigue_card = QFrame()
        fatigue_card.setObjectName("Card")
        fatigue_title = QLabel("rPPG 生理疲劳度")
        fatigue_title.setObjectName("SectionTitle")
        self.fatigue_label = QLabel("--")
        self.fatigue_label.setAlignment(Qt.AlignCenter)
        self.fatigue_label.setStyleSheet(
            f"color: {MUTED}; font-size: 74px; font-weight: 800; "
            "background: transparent; font-family: 'Courier New';"
        )
        fatigue_unit = QLabel("/ 100")
        fatigue_unit.setAlignment(Qt.AlignCenter)
        fatigue_unit.setStyleSheet(f"color: {MUTED}; font-size: 18px;")
        self.fatigue_status_label = QLabel("等待有效生理数据")
        self.fatigue_status_label.setAlignment(Qt.AlignCenter)
        self.fatigue_status_label.setWordWrap(True)
        self.fatigue_status_label.setStyleSheet(
            f"color: {MUTED}; background: {PLOT}; border-radius: 12px; "
            "padding: 12px; font-size: 15px;"
        )

        fatigue_layout = QVBoxLayout(fatigue_card)
        fatigue_layout.setContentsMargins(20, 18, 20, 20)
        fatigue_layout.addWidget(fatigue_title)
        fatigue_layout.addStretch(1)
        fatigue_layout.addWidget(self.fatigue_label)
        fatigue_layout.addWidget(fatigue_unit)
        fatigue_layout.addSpacing(18)
        fatigue_layout.addWidget(self.fatigue_status_label)
        fatigue_layout.addStretch(1)

        main_row = QHBoxLayout()
        main_row.setSpacing(16)
        main_row.addWidget(wave_card, 6)
        main_row.addWidget(bpm_card, 3)
        main_row.addWidget(fatigue_card, 3)

        self.long_card = MetricCard("Long BPM", "--", "最近约 8～12 秒")
        self.instant_card = MetricCard("Instant BPM", "--", "当前约 4.23 秒窗口")
        self.quality_card = MetricCard("数据质量", "等待中", "等待上位机连接")
        self.confidence_card = MetricCard("可信度", "--", "长窗口频谱可信度")

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)
        metrics.setVerticalSpacing(14)
        metrics.addWidget(self.long_card, 0, 0)
        metrics.addWidget(self.instant_card, 0, 1)
        metrics.addWidget(self.quality_card, 0, 2)
        metrics.addWidget(self.confidence_card, 0, 3)

        status_card = QFrame()
        status_card.setObjectName("Card")
        status_title = QLabel("运行状态")
        status_title.setObjectName("SectionTitle")
        self.status_label = QLabel("接口已预留，当前未启动数据接收")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        self.detail_label = QLabel(
            "后续接收端只需调用 RPPGWidget.update_rppg_data(payload)，"
            "或 FusionState.update_rppg(payload)。"
        )
        self.detail_label.setObjectName("MutedLabel")
        self.detail_label.setWordWrap(True)

        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(18, 14, 18, 14)
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.detail_label)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(16)
        root.addLayout(header)
        root.addLayout(main_row, 1)
        root.addLayout(metrics)
        root.addWidget(status_card)

    def update_rppg_data(self, payload):
        """预留的线程安全数据入口，供后续通信模块直接调用。"""
        self.external_data.emit(dict(payload))

    def _accept_external_data(self, payload):
        # 每收到一帧新数据，在 RK3588 本地执行一次轻量疲劳计算。
        enriched_payload = dict(payload)
        enriched_payload.update(self.fatigue_estimator.update(enriched_payload))
        FusionState.update_rppg(enriched_payload)
        self.refresh_ui()

    @staticmethod
    def _fmt_bpm(value):
        return "--" if value is None else f"{value:.1f}"

    @staticmethod
    def _interpret_bpm(bpm, quality):
        if bpm is None:
            return "等待有效 rPPG 数据", MUTED
        if quality not in ("good", "medium"):
            return "当前数据质量不足，暂不解读心率", WARNING
        # 仅用于驾驶状态界面提示，不作为医学诊断。
        if bpm < 50:
            return "心率偏低，建议结合个人基线观察", WARNING
        if bpm <= 100:
            return "心率处于常见静息范围，当前较稳定", SUCCESS
        if bpm <= 120:
            return "心率有所升高，请结合紧张、动作等因素判断", WARNING
        return "心率明显升高，建议持续观察并确认数据质量", DANGER

    def refresh_ui(self):
        data = FusionState.get_rppg_snapshot()
        received_at = data["received_at"]
        fresh = received_at is not None and (time.time() - received_at) <= 3.0
        connected = bool(data["connected"] and fresh)

        if connected:
            self.connection_label.setText("●  已接收数据")
            self.connection_label.setStyleSheet(
                f"color: {SUCCESS}; background: {PLOT}; border: 1px solid {SUCCESS}; "
                "border-radius: 12px; padding: 9px 16px; font-size: 14px; font-weight: 600;"
            )
        elif data["connected"]:
            self.connection_label.setText("●  数据已超时")
            self.connection_label.setStyleSheet(
                f"color: {WARNING}; background: {PLOT}; border: 1px solid {WARNING}; "
                "border-radius: 12px; padding: 9px 16px; font-size: 14px; font-weight: 600;"
            )
        else:
            self.connection_label.setText("●  等待接入")
            self.connection_label.setStyleSheet(
                f"color: {MUTED}; background: {PLOT}; border: 1px solid {BORDER}; "
                "border-radius: 12px; padding: 9px 16px; font-size: 14px; font-weight: 600;"
            )

        bpm = data["bpm"]
        quality = data["quality"]
        quality_text = self.QUALITY_TEXT.get(quality, quality or "未知")
        quality_color = self.QUALITY_COLOR.get(quality, MUTED)

        self.bpm_label.setText(self._fmt_bpm(bpm))
        interpretation, interpretation_color = self._interpret_bpm(bpm, quality)
        self.interpretation_label.setText(interpretation)
        self.interpretation_label.setStyleSheet(
            f"color: {interpretation_color}; background: {PLOT}; border-radius: 12px; "
            "padding: 12px; font-size: 16px; font-weight: 600;"
        )

        fatigue_score = data.get("rppg_fatigue_score")
        calibrated = bool(data.get("rppg_calibrated", False))
        progress = float(data.get("rppg_baseline_progress", 0.0) or 0.0)
        fatigue_status = str(
            data.get("rppg_fatigue_status") or "等待有效生理数据"
        )

        if fatigue_score is None:
            fatigue_color = MUTED
            self.fatigue_label.setText("--")
            self.fatigue_status_label.setText(fatigue_status)
        elif not calibrated:
            fatigue_color = WARNING
            self.fatigue_label.setText(f"{fatigue_score:.1f}")
            self.fatigue_status_label.setText(
                f"{fatigue_status}\n基线完成度：{progress * 100:.0f}%"
            )
        else:
            self.fatigue_label.setText(f"{fatigue_score:.1f}")
            if fatigue_score < 35.0:
                fatigue_color = SUCCESS
                level_text = "生理状态良好（清醒）"
            elif fatigue_score < 70.0:
                fatigue_color = WARNING
                level_text = "警告：出现轻度生理疲劳"
            else:
                fatigue_color = DANGER
                level_text = "危险：出现深度生理疲劳"
            self.fatigue_status_label.setText(
                f"{level_text}\n{fatigue_status}"
            )

        self.fatigue_label.setStyleSheet(
            f"color: {fatigue_color}; font-size: 74px; font-weight: 800; "
            "background: transparent; font-family: 'Courier New';"
        )
        self.fatigue_status_label.setStyleSheet(
            f"color: {fatigue_color}; background: {PLOT}; border-radius: 12px; "
            "padding: 12px; font-size: 15px; font-weight: 600;"
        )

        self.long_card.set_value(self._fmt_bpm(data["long_bpm"]))
        self.instant_card.set_value(self._fmt_bpm(data["instant_bpm"]))
        self.quality_card.set_value(
            quality_text,
            data["quality_reason"],
            quality_color,
        )
        confidence = data["confidence"]
        self.confidence_card.set_value(
            "--" if bpm is None else f"{confidence * 100:.0f}%",
            "长窗口频谱可信度",
            ACCENT if bpm is not None else MUTED,
        )

        self.wave_widget.set_signal(data["signal"], data["sample_rate"])

        fps_text = "--" if data["fps"] is None else f"{data['fps']:.1f}"
        self.status_label.setText(
            f"状态：{data['status']}    |    跟踪：{data['tracker_status']}"
        )
        rmssd_current = data.get("rppg_rmssd_current")
        rmssd_base = data.get("rppg_rmssd_base")
        hr_base = data.get("rppg_hr_base")
        ppi_count = int(data.get("rppg_ppi_count", 0) or 0)
        bvp_seconds = float(data.get("rppg_bvp_seconds", 0.0) or 0.0)
        rmssd_text = "--" if rmssd_current is None else f"{rmssd_current:.1f} ms"
        rmssd_base_text = "--" if rmssd_base is None else f"{rmssd_base:.1f} ms"
        hr_base_text = "--" if hr_base is None else f"{hr_base:.1f} BPM"

        self.detail_label.setText(
            f"窗口 FPS：{fps_text}    |    BVP缓冲：{bvp_seconds:.1f}/60秒    |    "
            f"PPI：{ppi_count}个    |    当前RMSSD：{rmssd_text}\n"
            f"HR基线：{hr_base_text}    |    RMSSD基线：{rmssd_base_text}    |    "
            f"评分采样率：{RPPGFatigueEstimator.TARGET_FS:.1f} Hz"
        )
