import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui_theme import ACCENT, APP_STYLESHEET, MUTED, TEXT
from fusion_state import FusionState
from rppg_receiver import RPPGReceiver

# 运行 server_physmamba.py 的电脑局域网 IP。
# 默认沿用眼动模块中的电脑地址；地址变化时只改这一行即可。
# 也可以启动前设置环境变量：export RPPG_SERVER_IP=192.168.1.100
RPPG_SERVER_IP = os.environ.get("RPPG_SERVER_IP", "10.69.27.97")
RPPG_WS_URI = f"ws://{RPPG_SERVER_IP}:8000/rppg_ws"


class HomeWidget(QWidget):
    def __init__(self):
        super().__init__()

        badge = QLabel("RK3588 · 多模态边缘智能")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"color: {ACCENT}; background: #082f49; border-radius: 13px; "
            "padding: 7px 14px; font-size: 15px; font-weight: 600;"
        )

        title = QLabel("驾驶员多模态状态监测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 46px; font-weight: 800; background: transparent;"
        )

        subtitle = QLabel(
            "眼动疲劳分析  ·  rPPG 生理状态  ·  危险行为检测  ·  综合风险判断"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {MUTED}; font-size: 20px; background: transparent;"
        )

        hint = QLabel("请从左侧选择功能模块")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(
            f"color: {MUTED}; font-size: 16px; background: transparent; margin-top: 20px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.addStretch(2)
        layout.addWidget(badge, 0, Qt.AlignHCenter)
        layout.addSpacing(24)
        layout.addWidget(title)
        layout.addSpacing(18)
        layout.addWidget(subtitle)
        layout.addWidget(hint)
        layout.addStretch(3)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle("驾驶员多模态状态监测系统")
        self.setStyleSheet(APP_STYLESHEET)

        screen = QApplication.primaryScreen().availableGeometry()
        width = min(1680, max(1180, screen.width() - 70))
        height = min(980, max(720, screen.height() - 70))
        self.resize(width, height)
        self.setMinimumSize(1100, 680)

        from eye_widget import EyeTrackingWidget
        from rppg_widget import RPPGWidget
        from yolo_widget import YoloWidget
        from fusion_widget import FusionWidget

        self.stack = QStackedWidget()
        self.home_widget = HomeWidget()
        self.eye_widget = EyeTrackingWidget()
        self.rppg_widget = RPPGWidget()
        self.yolo_widget = YoloWidget()
        self.fusion_widget = FusionWidget()

        # 接收电脑端 rPPG JSON，并交给你已有的线程安全数据入口。
        self.rppg_receiver = RPPGReceiver(RPPG_WS_URI, self)
        self.rppg_receiver.data_received.connect(
            self.rppg_widget.update_rppg_data
        )
        self.rppg_receiver.connection_changed.connect(
            self._on_rppg_connection_changed
        )
        self.rppg_receiver.start()

        self.stack.addWidget(self.home_widget)       # 0
        self.stack.addWidget(self.eye_widget)        # 1
        self.stack.addWidget(self.rppg_widget)       # 2
        self.stack.addWidget(self.yolo_widget)       # 3
        self.stack.addWidget(self.fusion_widget)     # 4

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(245)

        brand = QLabel("DRIVER\nMONITOR")
        brand.setStyleSheet(
            f"color: {TEXT}; font-size: 24px; font-weight: 800; "
            f"padding: 8px 16px; background: transparent;"
        )
        brand_subtitle = QLabel("RK3588 智能终端")
        brand_subtitle.setStyleSheet(
            f"color: {ACCENT}; font-size: 14px; padding: 0 16px; background: transparent;"
        )

        nav_items = [
            ("首页", 0),
            ("眼动监测", 1),
            ("rPPG 监测", 2),
            ("行为检测", 3),
            ("综合判断", 4),
        ]
        self.nav_buttons = []
        nav_layout = QVBoxLayout(sidebar)
        nav_layout.setContentsMargins(12, 22, 12, 18)
        nav_layout.setSpacing(4)
        nav_layout.addWidget(brand)
        nav_layout.addWidget(brand_subtitle)
        nav_layout.addSpacing(30)

        for text, index in nav_items:
            button = QPushButton(text)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=index: self.display(i))
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)

        nav_layout.addStretch(1)
        footer = QLabel("多模态感知 · 本地实时显示")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f"color: {MUTED}; font-size: 12px; padding: 10px; background: transparent;"
        )
        nav_layout.addWidget(footer)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)
        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)

        self.display(0)

    def display(self, index):
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self.stack.setCurrentIndex(index)

    def _on_rppg_connection_changed(self, message, connected):
        if not connected:
            # 清除旧数据，避免综合判断继续使用超时心率。
            FusionState.clear_rppg(message)

    def closeEvent(self, event):
        if hasattr(self, "rppg_receiver"):
            self.rppg_receiver.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
