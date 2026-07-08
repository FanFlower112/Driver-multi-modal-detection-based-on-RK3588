# main_gui.py

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QHBoxLayout,
    QVBoxLayout, QStackedWidget
)
from PyQt5.QtCore import Qt

from PyQt5.QtGui import QFont, QPalette, QColor   # 添加字体颜色支持

from eeg_widget import EEGDisplayWidget
from eye_widget import EyeTrackingWidget
from yolo_widget import YoloWidget
from fusion_widget import FusionWidget


# 模块窗口示例（你后续可替换为实际显示脑电/眼动/YOLO的功能窗口）



class HomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        # 创建大字体标签
        self.label = QLabel("基于RK3588的驾驶分心检测系统")
        self.label.setAlignment(Qt.AlignCenter)
        
        # 设置大字体 - 使用字体并加粗
        font = QFont()
        font.setPointSize(50)
        font.setBold(True)
        self.label.setFont(font)
        
        # 添加其他欢迎信息
        self.info_label = QLabel("欢迎使用驾驶分心检测系统\n请选择左侧功能模块开始监控")
        self.info_label.setAlignment(Qt.AlignCenter)
        info_font = QFont()
        info_font.setPointSize(20)
        self.info_label.setFont(info_font)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.addStretch(1)  # 顶部空白
        layout.addWidget(self.label)
        layout.addSpacing(50)  # 标题与描述之间的间距
        layout.addWidget(self.info_label)
        layout.addStretch(1)  # 底部空白
        
        self.setLayout(layout)

# 主窗口
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("驾驶分心检测系统")
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen().geometry()
        max_width = min(1920, screen.width() - 100)  # 留出100像素边距
        max_height = min(1080, screen.height() - 100)
        
        # 设置窗口最大尺寸
        self.setMaximumSize(max_width, max_height)
        
        # 设置16:9的初始窗口大小
        self.resize(1920, 1280)

        # 设置应用背景色
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 245, 250))  # 淡蓝色背景
        self.setPalette(palette)
        
        # 左侧按钮区
        self.button_home = QPushButton("首页")
        self.button_eeg = QPushButton("脑电监测")
        self.button_eye = QPushButton("眼动监测")
        self.button_yolo = QPushButton("YOLO检测")
        self.button_fusion = QPushButton("综合判断")
        
        # 设置按钮样式
        # 设置按钮样式 - 更大更清晰的按钮
        button_style = """
            QPushButton {
                font-size: 24px;
                padding: 20px;
                min-height: 60px;
                background-color: #e0e8f0;
                border: 2px solid #a0b0c0;
                border-radius: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #d0d8e0;
                border: 2px solid #8090a0;
            }
            QPushButton:pressed {
                background-color: #c0c8d0;
            }
            QPushButton:checked {
                background-color: #a0c0e0;
                font-weight: bold;
            }
        """
        for btn in [self.button_home, self.button_eeg, self.button_eye, 
                   self.button_yolo, self.button_fusion]:
            btn.setStyleSheet(button_style)
            btn.setCheckable(True)  # 使按钮可选中
        
        # 连接按钮信号
        self.button_home.clicked.connect(lambda: self.display(0))
        self.button_eeg.clicked.connect(lambda: self.display(1))
        self.button_eye.clicked.connect(lambda: self.display(2))
        self.button_yolo.clicked.connect(lambda: self.display(3))
        self.button_fusion.clicked.connect(lambda: self.display(4))
        
        button_layout = QVBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.button_home)        
        button_layout.addWidget(self.button_eeg)
        button_layout.addWidget(self.button_eye)
        button_layout.addWidget(self.button_yolo)
        button_layout.addWidget(self.button_fusion)
        button_layout.addStretch(1)
        
        # 右侧堆叠内容区
        self.stack = QStackedWidget()
        self.stack.addWidget(HomeWidget())  # 首页
        self.stack.addWidget(EEGDisplayWidget())  # EEG 实时显示页面
        self.stack.addWidget(EyeTrackingWidget())
        self.stack.addWidget(YoloWidget())
        self.stack.addWidget(FusionWidget())
        
        # 设置内容区域样式
        self.stack.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #c0c8d0;
                border-radius: 15px;
                margin: 20px;
            }
        """)
        
        # 总体布局
        layout = QHBoxLayout()
        layout.addLayout(button_layout, 1)  # 按钮区域占1份
        layout.addWidget(self.stack, 4)     # 内容区域占4份
        layout.setContentsMargins(30, 30, 30, 30)  # 设置外边距
        layout.setSpacing(30)  # 设置内部间距
        
        self.setLayout(layout)
        self.display(0)  # 默认显示首页
        self.button_home.setChecked(True)  # 标记首页按钮为选中状态

    def display(self, index):
        # 取消所有按钮的选中状态
            for btn in [self.button_home, self.button_eeg, self.button_eye, 
                    self.button_yolo, self.button_fusion]:
                btn.setChecked(False)
            
            # 设置当前按钮为选中状态
            if index == 0:
                self.button_home.setChecked(True)
            elif index == 1:
                self.button_eeg.setChecked(True)
            elif index == 2:
                self.button_eye.setChecked(True)
            elif index == 3:
                self.button_yolo.setChecked(True)
            elif index == 4:
                self.button_fusion.setChecked(True)
        
            self.stack.setCurrentIndex(index)

# 程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用整体样式
    app.setStyle("Fusion")  # 使用Fusion样式，更现代
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
