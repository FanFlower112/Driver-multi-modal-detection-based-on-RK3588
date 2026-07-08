# eye_widget.py

import asyncio
import json
import threading
import time
from collections import deque

import numpy as np
import websockets
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

from fusion_state import FusionState
from ui_theme import ACCENT, BORDER, DANGER, MUTED, PLOT, SUCCESS, TEXT, WARNING

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
        "edge_ratio": deque(maxlen=MAX_HISTORY),
    },
    "fatigue": deque(maxlen=MAX_HISTORY),
    "trajectory": deque(maxlen=MAX_HISTORY * 2),
}

INITIAL_WEIGHTS = {
    "perclos": 0.25,
    "blink_freq": 0.2,
    "avg_blink": 0.15,
    "fixation_rate": 0.15,
    "avg_fixation": 0.15,
    "edge_ratio": 0.1,
}


class EyeTrackingWidget(QWidget):
    connection_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.plot_counter = 0
        self.distraction_count = 0
        self.distraction_time = 0
        self.distraction_active = False
        self.distraction_start = None
        self.ws_connected = False
        self.connection_changed.connect(self._set_connection_status)

        self.init_ui()
        self.start_ws_thread()

    def init_ui(self):
        page_title = QLabel("眼动疲劳与分心监测")
        page_title.setObjectName("PageTitle")
        page_subtitle = QLabel("基于视线轨迹、眨眼与注视特征评估疲劳和持续分心")
        page_subtitle.setObjectName("PageSubtitle")

        self.connection_label = QLabel("●  正在连接眼动数据")
        self.connection_label.setAlignment(Qt.AlignCenter)
        self.connection_label.setStyleSheet(
            f"color: {WARNING}; background: {PLOT}; border: 1px solid {WARNING}; "
            "border-radius: 12px; padding: 9px 15px; font-size: 14px; font-weight: 600;"
        )

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(page_title)
        title_box.addWidget(page_subtitle)

        header = QHBoxLayout()
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(self.connection_label)

        plot_card = QFrame()
        plot_card.setObjectName("Card")
        self.fig, (self.ax_track, self.ax_score) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.patch.set_facecolor("#151e2e")
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.13, wspace=0.22)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {PLOT}; border-radius: 12px;")

        for axis in (self.ax_track, self.ax_score):
            axis.set_facecolor(PLOT)
            axis.tick_params(colors=MUTED, labelsize=10)
            axis.grid(True, color=BORDER, alpha=0.55, linewidth=0.8)
            for spine in axis.spines.values():
                spine.set_color(BORDER)

        self.ax_track.set_title("视线轨迹", fontsize=16, color=TEXT, pad=12)
        self.ax_track.set_xlim(0, 1)
        self.ax_track.set_ylim(0, 1)
        self.ax_track.set_xlabel("归一化 X", color=MUTED)
        self.ax_track.set_ylabel("归一化 Y", color=MUTED)
        self.scatter = self.ax_track.scatter([], [], s=24, alpha=0.75)
        self.focus_box = Rectangle(
            (0.1, 0.1),
            0.8,
            0.8,
            fill=True,
            facecolor="#0c4a6e",
            edgecolor=ACCENT,
            alpha=0.28,
            linestyle="--",
            linewidth=1.6,
        )
        self.ax_track.add_patch(self.focus_box)

        self.ax_score.set_title("眼动疲劳评分", fontsize=16, color=TEXT, pad=12)
        self.ax_score.set_ylim(0, 1.1)
        self.ax_score.set_xlabel("时间", color=MUTED)
        self.ax_score.set_ylabel("疲劳度", color=MUTED)
        self.fatigue_line, = self.ax_score.plot([], [], color=ACCENT, linewidth=2.0)

        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.addWidget(self.canvas)

        self.status_label = QLabel("状态：等待输入")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setMinimumHeight(86)

        self.focus_label = QLabel("当前未检测到持续分心")
        self.focus_label.setAlignment(Qt.AlignCenter)
        self.focus_label.setWordWrap(True)
        self.focus_label.setMinimumHeight(86)
        self.focus_label.setStyleSheet(
            f"font-size: 17px; color: {SUCCESS}; font-weight: 600; background: {PLOT}; "
            f"border: 1px solid {BORDER}; border-radius: 13px; padding: 12px;"
        )

        self.counter_label = QLabel("分心次数：0\n累计时长：0.0 秒")
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setMinimumHeight(86)
        self.counter_label.setStyleSheet(
            f"font-size: 17px; color: {TEXT}; font-weight: 600; background: {PLOT}; "
            f"border: 1px solid {BORDER}; border-radius: 13px; padding: 12px;"
        )

        status_row = QHBoxLayout()
        status_row.setSpacing(14)
        status_row.addWidget(self.status_label, 2)
        status_row.addWidget(self.focus_label, 2)
        status_row.addWidget(self.counter_label, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(16)
        root.addLayout(header)
        root.addWidget(plot_card, 1)
        root.addLayout(status_row)

        self.timer = QTimer(self)
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
            x, y, states = zip(*viz_data["trajectory"])
            color_map = {
                "blink": DANGER,
                "edge": WARNING,
                "fixation": SUCCESS,
            }
            self.scatter.set_offsets(np.column_stack((x, y)))
            self.scatter.set_color([color_map[state] for state in states])

            gx, gy = x[-1], y[-1]
            in_focus = 0.1 <= gx <= 0.9 and 0.1 <= gy <= 0.9
            now = time.time()

            if not in_focus:
                if not self.distraction_active:
                    self.distraction_active = True
                    self.distraction_start = now
                    self.distraction_count += 1
                elif now - self.distraction_start >= 3:
                    self.focus_label.setText("⚠ 持续分心超过 3 秒，请专注驾驶")
                    self.focus_label.setStyleSheet(
                        f"font-size: 17px; color: {DANGER}; font-weight: 700; background: {PLOT}; "
                        f"border: 1px solid {DANGER}; border-radius: 13px; padding: 12px;"
                    )
            else:
                if self.distraction_active:
                    self.distraction_time += now - self.distraction_start
                self.distraction_active = False
                self.focus_label.setText("当前未检测到持续分心")
                self.focus_label.setStyleSheet(
                    f"font-size: 17px; color: {SUCCESS}; font-weight: 600; background: {PLOT}; "
                    f"border: 1px solid {BORDER}; border-radius: 13px; padding: 12px;"
                )

            active_duration = 0.0
            if self.distraction_active and self.distraction_start is not None:
                active_duration = now - self.distraction_start
            self.counter_label.setText(
                f"分心次数：{self.distraction_count}\n"
                f"累计时长：{self.distraction_time + active_duration:.1f} 秒"
            )

        self.canvas.draw_idle()

        if fs:
            score = max(0.0, min(float(fs[-1]), 1.0))
            if score > 0.7:
                color, text = DANGER, "疲劳风险较高"
            elif score > 0.5:
                color, text = WARNING, "疲劳状态需要关注"
            else:
                color, text = SUCCESS, "眼动状态正常"
            self.status_label.setText(f"{text}\n疲劳得分：{score:.3f}")
            self.status_label.setStyleSheet(
                f"font-size: 19px; color: {color}; font-weight: 700; background: {PLOT}; "
                f"border: 1px solid {color}; border-radius: 13px; padding: 12px;"
            )
        else:
            self.status_label.setText("状态：等待眼动数据")
            self.status_label.setStyleSheet(
                f"font-size: 19px; color: {MUTED}; font-weight: 600; background: {PLOT}; "
                f"border: 1px solid {BORDER}; border-radius: 13px; padding: 12px;"
            )

    def _set_connection_status(self, state):
        if state == "connected":
            text, color = "●  眼动数据已连接", SUCCESS
        elif state == "failed":
            text, color = "●  眼动连接失败", DANGER
        else:
            text, color = "●  正在连接眼动数据", WARNING
        self.connection_label.setText(text)
        self.connection_label.setStyleSheet(
            f"color: {color}; background: {PLOT}; border: 1px solid {color}; "
            "border-radius: 12px; padding: 9px 15px; font-size: 14px; font-weight: 600;"
        )

    def start_ws_thread(self):
        thread = threading.Thread(target=self.ws_loop, daemon=True)
        thread.start()

    def ws_loop(self):
        asyncio.run(self.ws_main())

    async def ws_main(self):
        uri = "ws://10.69.27.97:6789"  # 后续可改为配置文件
        window = []
        window_ms = 60000
        try:
            async with websockets.connect(uri) as ws:
                self.ws_connected = True
                self.connection_changed.emit("connected")
                while True:
                    msg = await ws.recv()
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "location":
                            gaze = data.get("data")
                            if isinstance(gaze, list) and len(gaze) == 2:
                                x, y = map(float, gaze)
                                timestamp = time.time() * 1000
                                state = determine_state(x, y)
                                viz_data["trajectory"].append((x, y, state))
                                window.append({"timestamp": timestamp, "x": x, "y": y})
                                while window and timestamp - window[0]["timestamp"] > window_ms:
                                    window.pop(0)

                                proc = preprocess_window(window)
                                if proc and proc["window_duration"] > 0.1:
                                    perclos = proc["total_blink"] / proc["window_duration"]
                                    blink_f = (
                                        proc["blink_count"] / (proc["window_duration"] / 1000)
                                    ) * 60
                                    avg_blink = (
                                        (proc["total_blink"] / 1000) / proc["blink_count"]
                                        if proc["blink_count"]
                                        else 0
                                    )
                                    fix_rate = (
                                        proc["fixation_count"] / (proc["window_duration"] / 1000)
                                    ) * 60
                                    avg_fix = (
                                        (proc["total_fixation"] / 1000) / proc["fixation_count"]
                                        if proc["fixation_count"]
                                        else 0
                                    )
                                    edge = proc["total_edge"] / proc["window_duration"]
                                    norm = {
                                        "perclos": normalize_perclos(perclos),
                                        "blink_freq": normalize_blink_freq(blink_f),
                                        "avg_blink": normalize_avg_blink(avg_blink),
                                        "fixation_rate": normalize_fixation_rate(fix_rate),
                                        "avg_fixation": normalize_avg_fixation(avg_fix),
                                        "edge_ratio": normalize_edge_ratio(edge),
                                    }
                                    weights = calculate_dynamic_weights(norm)
                                    raw_score = fatigue_score(norm, weights) - 0.3
                                    score = max(0.0, min(float(raw_score), 1.0))
                                    FusionState.latest_eye_score = score

                                    now = time.time()
                                    viz_data["timestamps"].append(now)
                                    for key in norm:
                                        viz_data["features"][key].append(norm[key])
                                    viz_data["fatigue"].append(score)
                    except Exception as exc:
                        print("处理错误：", exc)
        except Exception as exc:
            self.ws_connected = False
            self.connection_changed.emit("failed")
            print("WebSocket连接失败：", exc)


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
