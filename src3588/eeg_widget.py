# eeg_widget.py

import asyncio
import websockets
import json
import numpy as np
import threading
from rknnlite.api import RKNNLite

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from fusion_state import FusionState


class EEGDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()

        # ===== 参数初始化 =====
        self.buffer_size = 1000
        self.eeg_buffer = np.zeros((8, self.buffer_size))
        self.window_buffer = np.zeros((0, 8))
        self.WINDOW_SIZE = 50
        self.data_received = False
        self.last_pred = "待输入"
        self.last_conf = 0.0

        # ===== 初始化模型推理器 =====
        self.processor = DataProcessor()
        self.inferencer = RKNNInferencer("eegLSTM_model_Q.rknn")

        # ===== 初始化界面 =====
        self.init_ui()

        # ===== 启动定时器更新图像 =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)

        # ===== 启动接收线程 =====
        threading.Thread(target=self.receive_data_loop, daemon=True).start()

    def init_ui(self):
        self.fig, self.axes = plt.subplots(8, 1, figsize=(6, 6))
        self.fig.suptitle("Eight-channel real-time EEG waveforms", fontsize=22, fontweight='bold')
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot(np.zeros(self.buffer_size))
            ax.set_ylim(-1000, 1000)
            ax.set_ylabel(f"Ch {i+1}", fontsize=14)
            ax.set_xticks([])
            self.lines.append(line)
        self.canvas = FigureCanvas(self.fig)

        self.status_label = QLabel("状态：待输入", self)
        self.status_label.setStyleSheet("font-size: 26px; color: navy;")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.conf_label = QLabel("置信度：-", self)
        self.conf_label.setStyleSheet("font-size: 22px; color: gray;")
        self.conf_label.setAlignment(Qt.AlignCenter)

        status_layout = QVBoxLayout()
        title = QLabel("清醒 / 疲劳判断", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        status_layout.addWidget(title)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.conf_label)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.canvas, 2)
        main_layout.addLayout(status_layout, 1)
        self.setLayout(main_layout)

    def update_plot(self):
        for ch in range(8):
            self.lines[ch].set_ydata(self.eeg_buffer[ch])
        self.canvas.draw()

        if self.data_received:
            if self.last_pred == 0:
                self.status_label.setText("状态：清醒")
                self.status_label.setStyleSheet("font-size: 36px; color: green; font-weight: bold;")
            else:
                self.status_label.setText("状态：疲劳")
                self.status_label.setStyleSheet("font-size: 36px; color: red; font-weight: bold;")
            self.conf_label.setText(f"置信度：{self.last_conf:.2f}")
        else:
            self.status_label.setText("状态：待输入")
            self.status_label.setStyleSheet("font-size: 36px; color: gray; font-weight: bold;")
            self.conf_label.setText("置信度：-")

    def receive_data_loop(self):
        asyncio.run(self.receive_data())

    async def receive_data(self):
        uri = "ws://192.168.1.100:8765"  # 请替换为实际地址
        try:
            async with websockets.connect(uri) as ws:
                print("[EEG] 已连接 WebSocket")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if isinstance(data, list) and len(data) == 8:
                        self.data_received = True
                        self.eeg_buffer[:, :-1] = self.eeg_buffer[:, 1:]
                        self.eeg_buffer[:, -1] = data

                        self.window_buffer = np.vstack([self.window_buffer, np.array(data).reshape(1, -1)])
                        if self.window_buffer.shape[0] >= self.WINDOW_SIZE:
                            window = self.window_buffer[-self.WINDOW_SIZE:]
                            input_tensor = self.processor.process(window)
                            pred, probs = self.inferencer.infer(input_tensor)
                            self.last_pred = pred
                            self.last_conf = probs[0][pred]  # 概率值 ∈ [0,1]

                            FusionState.latest_eeg_score = float(self.last_conf)
                            FusionState.eeg_pred_class = int(pred)

                            self.window_buffer = self.window_buffer[-(self.WINDOW_SIZE - 1):]
        except Exception as e:
            print("WebSocket连接失败：", e)


# ===== 数据标准化与推理类 =====
class DataProcessor:
    def __init__(self):
        self.train_mean = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
        self.train_std = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

    def fir_filter(self, data):
        return data

    def standardize(self, data):
        return (data - self.train_mean) / self.train_std

    def process(self, window_data):
        filtered = self.fir_filter(window_data)
        standardized = self.standardize(filtered)
        return standardized.astype(np.float32)


class RKNNInferencer:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def infer(self, input_data):
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)
        input_data = input_data.astype(np.float32)
        outputs = self.rknn.inference(inputs=[input_data])
        logits = outputs[0]
        probs = self.softmax(logits)
        pred = np.argmax(probs, axis=1)[0]
        return pred, probs

