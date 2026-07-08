# fusion_widget.py

import cv2
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap ,QFont
from fusion_state import FusionState

class FusionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.behavior_log = []
        self.init_ui()

        self.max_video_size = QSize(1280, 720)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(300)

    def init_ui(self):
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)

        data_font = QFont()
        data_font.setPointSize(20)

        alert_font = QFont()
        alert_font.setPointSize(22)
        alert_font.setBold(True)

        report_font = QFont()
        report_font.setPointSize(18)

        self.video_label = QLabel("等待视频流")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.eeg_label = QLabel("脑电疲劳度：--")
        self.eeg_label.setFont(data_font)

        self.eye_label = QLabel("眼动疲劳度：--")
        self.eye_label.setFont(data_font)

        self.fusion_label = QLabel("综合疲劳度：--")
        self.fusion_label.setFont(data_font)
        self.fusion_label.setStyleSheet("font-weight: bold;")

        self.status_label = QLabel("")
        self.status_label.setFont(alert_font)
        self.status_label.setAlignment(Qt.AlignCenter)

        behavior_title = QLabel("异常行为报告：")
        behavior_title.setFont(title_font)
        behavior_title.setStyleSheet("color: #333; margin-bottom: 5px;")

        self.behavior_box = QTextEdit()
        self.behavior_box.setReadOnly(True)
        self.behavior_box.setFont(report_font)
        self.behavior_box.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 8px;
            }
        """)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.eeg_label)
        left_layout.addWidget(self.eye_label)
        left_layout.addWidget(self.fusion_label)
        left_layout.addWidget(self.status_label)
        left_layout.addSpacing(5)
        left_layout.addWidget(behavior_title)
        left_layout.addWidget(self.behavior_box)
        left_layout.setSpacing(5)

        layout = QHBoxLayout()
        layout.addLayout(left_layout, 2)
        layout.addWidget(self.video_label, 3)
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)

    def update_gui(self):
        if FusionState.shared_yolo_frame is not None:
            rgb = cv2.cvtColor(FusionState.shared_yolo_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            label_size = self.video_label.size()
            scaled_size = label_size.boundedTo(self.max_video_size)
            scaled_pix = pix.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pix)
        else:
            self.video_label.setText("等待YOLO视频流...")

        eeg_score = FusionState.latest_eeg_score
        eye_score = FusionState.latest_eye_score
        eeg_pred_class = FusionState.eeg_pred_class

        if eeg_pred_class == 1:
            fusion_score = 0.6 * eeg_score + 0.4 * eye_score
        else:
            fusion_score = 0.4 * eye_score

        # ✅ 状态显示优化
        if eeg_pred_class == 1:
            self.eeg_label.setText(f"脑电疲劳度：{eeg_score:.2f}")
        else:
            self.eeg_label.setText("脑电疲劳度：0.00")

        self.eye_label.setText(f"眼动疲劳度：{eye_score:.2f}")
        self.fusion_label.setText(f"综合疲劳度：{fusion_score:.2f}")

        now_str = datetime.now().strftime("%H:%M:%S")

        if fusion_score > 0.8:
            self.status_label.setText("⚠ 检测到驾驶人疲劳\n请注意休息！")
            self.status_label.setStyleSheet("font-size: 30px; color: red; font-weight: bold;")
            self._log_event(now_str, "疲劳驾驶")
        else:
            self.status_label.setText("驾驶状态正常")
            self.status_label.setStyleSheet("font-size: 30px; color: green;")

        for behavior in FusionState.detected_behaviors:
            self._log_event(now_str, behavior)

        # ✅ 最新报告置顶
        report = "\n".join(f"[{t}] {msg}" for t, msg in reversed(self.behavior_log))
        self.behavior_box.setPlainText(report or "无")

    def _log_event(self, timestamp, event_text):
        if not self.behavior_log or self.behavior_log[-1][1] != event_text:
            self.behavior_log.append((timestamp, event_text))
