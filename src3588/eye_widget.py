# eye_widget.py

import asyncio
import websockets
import json
import time
from collections import deque
import numpy as np
import threading

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

from fusion_state import FusionState

# 可视化数据缓冲
MAX_HISTORY = 200
PLOT_INTERVAL = 5

viz_data = {
    "timestamps": deque(maxlen=MAX_HISTORY),
    "features": {
        "perclos": deque(maxlen=MAX_HISTORY),
        "blink_freq": deque(maxlen=MAX_HISTORY),
        "avg_blink": deque(maxlen=MAX_HISTORY),
        "fixation_rate": deque(maxlen=MAX_HISTORY),
        "avg_fixation": deque(maxlen=MAX_HISTORY),
        "edge_ratio": deque(maxlen=MAX_HISTORY)
    },
    "fatigue": deque(maxlen=MAX_HISTORY),
    "trajectory": deque(maxlen=MAX_HISTORY * 2)
}

INITIAL_WEIGHTS = {
    'perclos': 0.25,
    'blink_freq': 0.2,
    'avg_blink': 0.15,
    'fixation_rate': 0.15,
    'avg_fixation': 0.15,
    'edge_ratio': 0.1
}


class EyeTrackingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_counter = 0
        self.distraction_count = 0
        self.distraction_time = 0
        self.distraction_active = False
        self.distraction_start = None

        self.init_ui()
        self.start_ws_thread()

    def init_ui(self):
        font_big = QFont()
        font_big.setPointSize(20)
        font_big.setBold(True)

        self.fig, (self.ax_track, self.ax_score) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvas(self.fig)

        # Gaze轨迹图
        self.ax_track.set_title("Gaze Trajectory", fontsize=16)
        self.ax_track.set_xlim(0, 1)
        self.ax_track.set_ylim(0, 1)
        self.scatter = self.ax_track.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.6)

        # 添加专注区域：蓝色高亮框
        self.focus_box = Rectangle((0.1, 0.1), 0.8, 0.8,
                                   fill=True, facecolor='lightblue', edgecolor='blue', alpha=0.3, linestyle='--')
        self.ax_track.add_patch(self.focus_box)

        # 疲劳评分图
        self.ax_score.set_title("Fatigue Score", fontsize=16)
        self.ax_score.set_ylim(0, 1.1)
        self.fatigue_line, = self.ax_score.plot([], [], color="#e377c2")

        # 状态标签
        self.status_label = QLabel("状态：等待输入")
        self.status_label.setFont(font_big)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 分心提示
        self.focus_label = QLabel("")
        self.focus_label.setAlignment(Qt.AlignCenter)
        self.focus_label.setStyleSheet("font-size: 20px; color: red; font-weight: bold;")

        # 累计信息
        self.counter_label = QLabel("分心次数：0    累计时长：0.0 秒")
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setStyleSheet("font-size: 16px; color: #333;")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas, 7)
        layout.addWidget(self.status_label)
        layout.addWidget(self.focus_label)
        layout.addWidget(self.counter_label)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(500)

    def update_visualization(self):
        self.plot_counter += 1
        if self.plot_counter % PLOT_INTERVAL != 0:
            return

        ts = list(viz_data["timestamps"])
        fs = list(viz_data["fatigue"])

        self.fatigue_line.set_data(ts, fs)
        if len(ts) > 1:
            self.ax_score.set_xlim(ts[0], ts[-1])

        if viz_data["trajectory"]:
            x, y, s = zip(*viz_data["trajectory"])
            colors = [dict(blink='#d62728', edge='#8c564b', fixation='#2ca02c')[i] for i in s]
            self.scatter.set_offsets(np.column_stack((x, y)))
            self.scatter.set_color(colors)

            # 获取最新 gaze 点
            gx, gy = x[-1], y[-1]
            in_focus = 0.1 <= gx <= 0.9 and 0.1 <= gy <= 0.9
            now = time.time()

            if not in_focus:
                if not self.distraction_active:
                    self.distraction_active = True
                    self.distraction_start = now
                    self.distraction_count += 1
                elif now - self.distraction_start >= 3:
                    self.focus_label.setText("⚠ 持续分心超过3秒，请专注驾驶！")
            else:
                if self.distraction_active:
                    duration = now - self.distraction_start
                    self.distraction_time += duration
                self.distraction_active = False
                self.focus_label.setText("")

            self.counter_label.setText(
                f"分心次数：{self.distraction_count}    累计时长：{self.distraction_time:.1f} 秒")

        self.canvas.draw()

        # 状态栏颜色跟随疲劳程度
        if fs:
            score = fs[-1]
            self.status_label.setText(f"疲劳得分: {score:.3f}")
            if score > 0.7:
                self.status_label.setStyleSheet("font-size: 22px; color: white; background-color: #ff4d4d;")
            elif score > 0.5:
                self.status_label.setStyleSheet("font-size: 22px; color: white; background-color: #ffa500;")
            else:
                self.status_label.setStyleSheet("font-size: 22px; color: white; background-color: #66cc66;")
        else:
            self.status_label.setText("状态：等待输入")
            self.status_label.setStyleSheet("font-size: 22px; color: #333; background-color: #f0f0f0;")

    def start_ws_thread(self):
        thread = threading.Thread(target=self.ws_loop, daemon=True)
        thread.start()

    def ws_loop(self):
        asyncio.run(self.ws_main())

    async def ws_main(self):
        uri = "ws://192.168.1.100:6789"  # 替换为你的地址 192.168.1.100:6789
        window = []
        W_MS = 60000
        try:
            async with websockets.connect(uri) as ws:
                while True:
                    msg = await ws.recv()
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "location":
                            gaze = data.get("data")
                            if isinstance(gaze, list) and len(gaze) == 2:
                                x, y = map(float, gaze)
                                t = time.time() * 1000
                                state = determine_state(x, y)
                                viz_data["trajectory"].append((x, y, state))
                                window.append({'timestamp': t, 'x': x, 'y': y})
                                while window and (t - window[0]['timestamp'] > W_MS):
                                    window.pop(0)

                                proc = preprocess_window(window)
                                if proc:
                                    perclos = proc['total_blink'] / proc['window_duration']
                                    blink_f = (proc['blink_count'] / (proc['window_duration']/1000)) * 60
                                    avg_blink = (proc['total_blink']/1000) / proc['blink_count'] if proc['blink_count'] else 0
                                    fix_rate = (proc['fixation_count'] / (proc['window_duration']/1000)) * 60
                                    avg_fix = (proc['total_fixation']/1000) / proc['fixation_count'] if proc['fixation_count'] else 0
                                    edge = proc['total_edge'] / proc['window_duration']
                                    norm = {
                                        'perclos': normalize_perclos(perclos),
                                        'blink_freq': normalize_blink_freq(blink_f),
                                        'avg_blink': normalize_avg_blink(avg_blink),
                                        'fixation_rate': normalize_fixation_rate(fix_rate),
                                        'avg_fixation': normalize_avg_fixation(avg_fix),
                                        'edge_ratio': normalize_edge_ratio(edge)
                                    }
                                    w = calculate_dynamic_weights(norm)
                                    score = fatigue_score(norm, w)-0.3

                                    #FusionState.latest_eye_score = float(score)
                                    FusionState.latest_eye_score = max(0.0, min(float(score), 1.0))

                                    now = time.time()
                                    viz_data["timestamps"].append(now)
                                    for k in norm:
                                        viz_data["features"][k].append(norm[k])
                                    viz_data["fatigue"].append(score)
                    except Exception as e:
                        print("处理错误：", e)
        except Exception as e:
            print("WebSocket连接失败：", e)


# ========== 特征计算函数 ==========
def normalize_perclos(val): return min(val / 0.15, 1.0)
def normalize_blink_freq(val): return 1.0 if val < 10 or val > 35 else max(0, min((15-val)/5, 1)) if val < 15 else (val-30)/5
def normalize_avg_blink(val): return 0 if val <= 0.15 else min((val-0.15)/0.05, 1.0)
def normalize_fixation_rate(val): return 0 if val >= 80 else min((80 - val)/10, 1.0)
def normalize_avg_fixation(val): return 0 if val <= 0.65 else min((val-0.65)/0.05, 1.0)
def normalize_edge_ratio(val): return 0 if val <= 0.15 else min((val - 0.15)/0.05, 1.0)

def calculate_dynamic_weights(norm):
    base = {k: INITIAL_WEIGHTS[k] * (norm[k]+1e-5) for k in norm}
    total_base = sum(base.values())
    norm_w = {k: v/total_base for k, v in base.items()}
    constrained = {
        k: max(min(norm_w[k], INITIAL_WEIGHTS[k]*1.3), INITIAL_WEIGHTS[k]*0.7)
        for k in norm
    }
    total_con = sum(constrained.values())
    return {k: v/total_con for k, v in constrained.items()}

def fatigue_score(norm, w): return sum(norm[k] * w[k] for k in norm)

def determine_state(x, y):
    if x == 0 and y == 0: return 'blink'
    if x < 0 or x > 1 or y < 0 or y > 1: return 'edge'
    return 'fixation'

def preprocess_window(window):
    if not window: return None
    start, end = window[0]['timestamp'], window[-1]['timestamp']
    dur = end - start
    blink, edge, fix_total, blink_c, fix_c = 0, 0, 0, 0, 0
    state, s_time, fix_start, fx, fy = None, None, None, None, None

    for i, p in enumerate(window):
        x, y, t = p['x'], p['y'], p['timestamp']
        new_state = determine_state(x, y)

        if new_state != state:
            if state == 'blink': blink += t - s_time; blink_c += 1
            elif state == 'edge': edge += t - s_time
            elif state == 'fixation' and fix_start:
                d = t - fix_start
                if d >= 100: fix_total += d; fix_c += 1
            if new_state == 'fixation': fix_start, fx, fy = t, x, y
            else: fix_start = None
            state, s_time = new_state, t
        elif state == 'fixation':
            if abs(x-fx) > 0.04 or abs(y-fy) > 0.04:
                d = t - fix_start
                if d >= 100: fix_total += d; fix_c += 1
                fix_start, fx, fy = t, x, y

        if i == len(window) - 1:
            if state == 'blink': blink += t - s_time; blink_c += 1
            elif state == 'edge': edge += t - s_time
            elif state == 'fixation' and fix_start:
                d = t - fix_start
                if d >= 100: fix_total += d; fix_c += 1

    return {
        'window_duration': dur,
        'total_blink': blink,
        'blink_count': blink_c,
        'total_edge': edge,
        'total_fixation': fix_total,
        'fixation_count': fix_c
    }
