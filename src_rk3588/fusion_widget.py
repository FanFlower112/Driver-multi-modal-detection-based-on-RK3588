import time
from datetime import datetime

import cv2
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fusion_state import FusionState
from ui_theme import ACCENT, BORDER, DANGER, MUTED, PLOT, SUCCESS, TEXT, WARNING


class StatusCard(QFrame):
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
        layout.setContentsMargins(15, 13, 15, 13)
        layout.setSpacing(5)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def update_value(self, value, detail, color=TEXT):
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color: {color};")
        self.detail_label.setText(detail)


class FusionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.behavior_log = []
        self.max_video_size = QSize(1280, 720)
        self._last_rppg_alert = None
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(300)

    def _build_ui(self):
        page_title = QLabel("多模态综合判断")
        page_title.setObjectName("PageTitle")
        page_subtitle = QLabel("融合眼动疲劳、rPPG 生理状态与危险驾驶行为")
        page_subtitle.setObjectName("PageSubtitle")

        header = QVBoxLayout()
        header.setSpacing(4)
        header.addWidget(page_title)
        header.addWidget(page_subtitle)

        self.eye_card = StatusCard("眼动疲劳度", "0.00", "等待眼动数据")
        self.rppg_card = StatusCard("稳定心率", "--", "等待 rPPG 数据")
        self.quality_card = StatusCard("rPPG 质量", "等待中", "尚未接入")
        self.risk_card = StatusCard("综合驾驶风险", "低", "当前未发现明显风险")

        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(12)
        card_grid.setVerticalSpacing(12)
        card_grid.addWidget(self.eye_card, 0, 0)
        card_grid.addWidget(self.rppg_card, 0, 1)
        card_grid.addWidget(self.quality_card, 1, 0)
        card_grid.addWidget(self.risk_card, 1, 1)

        self.status_label = QLabel("驾驶状态正常")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(92)

        report_title = QLabel("事件与告警记录")
        report_title.setObjectName("SectionTitle")
        self.behavior_box = QTextEdit()
        self.behavior_box.setReadOnly(True)
        self.behavior_box.setPlaceholderText("暂无异常事件")

        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(14)
        left_layout.addLayout(card_grid)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(report_title)
        left_layout.addWidget(self.behavior_box, 1)

        video_card = QFrame()
        video_card.setObjectName("Card")
        video_title = QLabel("危险行为检测画面")
        video_title.setObjectName("SectionTitle")
        self.video_label = QLabel("等待行为检测视频流…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            f"background-color: {PLOT}; color: {MUTED}; border: 1px solid {BORDER}; "
            "border-radius: 12px; font-size: 17px;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(640, 480)

        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(12)
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video_label, 1)

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(left_card, 4)
        content.addWidget(video_card, 6)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(16)
        root.addLayout(header)
        root.addLayout(content, 1)

    @staticmethod
    def _quality_display(quality):
        mapping = {
            "good": ("良好", SUCCESS),
            "medium": ("一般", WARNING),
            "poor": ("较差", DANGER),
            "warming": ("预热中", MUTED),
            "waiting": ("等待中", MUTED),
        }
        return mapping.get(quality, (quality or "未知", MUTED))

    @staticmethod
    def _behavior_text(name):
        return {
            "phone": "检测到驾驶员使用手机",
            "smoke": "检测到驾驶员吸烟",
            "face": "检测到人脸",
        }.get(name, str(name))

    @staticmethod
    def _rppg_warning(bpm, quality):
        if bpm is None or quality not in ("good", "medium"):
            return None
        if bpm < 50:
            return "心率偏低"
        if bpm > 120:
            return "心率明显升高"
        if bpm > 100:
            return "心率有所升高"
        return None

    def update_gui(self):
        if FusionState.shared_yolo_frame is not None:
            rgb = cv2.cvtColor(FusionState.shared_yolo_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(image)
            target = self.video_label.size().boundedTo(self.max_video_size)
            self.video_label.setPixmap(
                pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.video_label.setText("等待行为检测视频流…")

        eye_score = max(0.0, min(float(FusionState.latest_eye_score), 1.0))
        behaviors = list(FusionState.detected_behaviors)
        rppg = FusionState.get_rppg_snapshot()

        received_at = rppg["received_at"]
        rppg_fresh = received_at is not None and time.time() - received_at <= 3.0
        bpm = rppg["bpm"] if rppg_fresh else None
        quality = rppg["quality"] if rppg_fresh else "waiting"
        quality_text, quality_color = self._quality_display(quality)

        eye_color = SUCCESS if eye_score < 0.5 else WARNING if eye_score < 0.7 else DANGER
        self.eye_card.update_value(
            f"{eye_score:.2f}",
            "正常" if eye_score < 0.5 else "需要关注" if eye_score < 0.7 else "疲劳风险较高",
            eye_color,
        )
        self.rppg_card.update_value(
            "--" if bpm is None else f"{bpm:.1f} BPM",
            "等待有效数据" if bpm is None else "稳定心率输出",
            ACCENT if bpm is not None else MUTED,
        )
        self.quality_card.update_value(
            quality_text,
            rppg["quality_reason"] if rppg_fresh else "尚未接入或数据已超时",
            quality_color,
        )

        # 综合风险采用规则判断：眼动为疲劳主依据，YOLO 与可靠 rPPG 异常用于提升风险等级。
        risk_score = eye_score
        reasons = []
        if eye_score >= 0.7:
            reasons.append("眼动疲劳度较高")
        elif eye_score >= 0.5:
            reasons.append("眼动疲劳度有所升高")

        if behaviors:
            risk_score = max(risk_score, 0.90)
            reasons.extend(self._behavior_text(item) for item in behaviors)

        rppg_warning = self._rppg_warning(bpm, quality)
        if rppg_warning:
            risk_score = max(risk_score, 0.72 if bpm is not None and bpm > 120 else 0.58)
            reasons.append(rppg_warning)

        if risk_score >= 0.8:
            risk_text, risk_color = "高", DANGER
            banner = "⚠ 检测到较高驾驶风险，请及时采取安全措施"
        elif risk_score >= 0.5:
            risk_text, risk_color = "中", WARNING
            banner = "驾驶状态需要关注，请保持专注并持续观察"
        else:
            risk_text, risk_color = "低", SUCCESS
            banner = "驾驶状态正常"

        detail = "；".join(dict.fromkeys(reasons)) if reasons else "当前未发现明显风险"
        self.risk_card.update_value(risk_text, detail, risk_color)
        self.status_label.setText(banner)
        self.status_label.setStyleSheet(
            f"color: {risk_color}; background: {PLOT}; border: 1px solid {risk_color}; "
            "border-radius: 14px; padding: 14px; font-size: 23px; font-weight: 700;"
        )

        now_str = datetime.now().strftime("%H:%M:%S")
        if eye_score >= 0.7:
            self._log_event(now_str, "眼动疲劳风险较高")
        for behavior in behaviors:
            self._log_event(now_str, self._behavior_text(behavior))
        if rppg_warning and rppg_warning != self._last_rppg_alert:
            self._log_event(now_str, rppg_warning)
        self._last_rppg_alert = rppg_warning

        report = "\n".join(f"[{t}] {msg}" for t, msg in reversed(self.behavior_log[-100:]))
        self.behavior_box.setPlainText(report or "暂无异常事件")

    def _log_event(self, timestamp, event_text):
        if not self.behavior_log or self.behavior_log[-1][1] != event_text:
            self.behavior_log.append((timestamp, event_text))
