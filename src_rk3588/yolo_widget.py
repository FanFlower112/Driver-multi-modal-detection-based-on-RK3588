# yolo_widget.py

import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QFrame
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from rknnlite.api import RKNNLite
from fusion_state import FusionState
from ui_theme import BORDER, DANGER, MUTED, PLOT, SUCCESS, TEXT

# 配置项
MODEL_PATH = "bests.rknn"
CLASS_NAMES = ["smoke", "face", "phone"]
ALERT_CLASSES = ["phone", "smoke"]
CLASS_COLORS = {
    "smoke": (255, 0, 0),  # 红色
    "face": (0, 255, 0),   # 绿色
    "phone": (0, 0, 255)   # 蓝色
}
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
FPS_LIMIT = 30

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def process(input, mask, anchors, obj_thresh):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = input.shape[:2]
    box_confidence = np.expand_dims(input[..., 4], axis=-1)
    box_class_probs = input[..., 5:]

    box_xy = input[..., :2] * 2 - 0.5
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy *= int(IMG_SIZE / grid_h)
    box_wh = np.square(input[..., 2:4] * 2)
    box_wh *= anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)
    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= obj_thresh)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= obj_thresh)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores, nms_thresh):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def yolov5_post_process(input_data, obj_thresh, nms_thresh):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23],
               [30, 61], [62, 45], [59, 119],
               [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors, obj_thresh)
        b, c, s = filter_boxes(b, c, s, obj_thresh)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    valid = [i for i, box_array in enumerate(boxes) if box_array.size > 0]
    if not valid:
        return None, None, None

    boxes = np.concatenate([boxes[i] for i in valid])
    classes = np.concatenate([classes[i] for i in valid])
    scores = np.concatenate([scores[i] for i in valid])
    boxes = xywh2xyxy(boxes)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_ = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s, nms_thresh)
        nboxes.append(b[keep])
        nclasses.append(c_[keep])
        nscores.append(s[keep])

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

class YoloWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.rknn = RKNNLite()
        assert self.rknn.load_rknn(MODEL_PATH) == 0
        assert self.rknn.init_runtime() == 0

        self.cap = cv2.VideoCapture("/dev/video21")
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头 /dev/video21")

        self._build_ui()
        self.last_alert_text = "无危险行为"

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / FPS_LIMIT))

    def _build_ui(self):
        page_title = QLabel("危险驾驶行为检测")
        page_title.setObjectName("PageTitle")
        page_subtitle = QLabel("RK3588 NPU 实时检测使用手机、吸烟与人脸目标")
        page_subtitle.setObjectName("PageSubtitle")

        header = QVBoxLayout()
        header.setSpacing(4)
        header.addWidget(page_title)
        header.addWidget(page_subtitle)

        video_card = QFrame()
        video_card.setObjectName("Card")
        self.video_label = QLabel("视频加载中…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            f"background-color: {PLOT}; color: {MUTED}; border: 1px solid {BORDER}; "
            "border-radius: 12px; font-size: 17px;"
        )
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.addWidget(self.video_label)

        info_card = QFrame()
        info_card.setObjectName("Card")
        info_title = QLabel("实时告警")
        info_title.setObjectName("SectionTitle")

        self.alert_label = QLabel("当前未检测到危险行为")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setWordWrap(True)
        self.alert_label.setMinimumHeight(130)
        self.alert_label.setStyleSheet(
            f"color: {SUCCESS}; background: {PLOT}; border: 1px solid {SUCCESS}; "
            "border-radius: 14px; padding: 16px; font-size: 21px; font-weight: 700;"
        )

        class_title = QLabel("检测类别")
        class_title.setObjectName("SectionTitle")
        class_info = QLabel("phone  使用手机\nsmoke  吸烟\nface  人脸")
        class_info.setStyleSheet(
            f"color: {TEXT}; background: {PLOT}; border: 1px solid {BORDER}; "
            "border-radius: 12px; padding: 16px; font-size: 17px; line-height: 1.8;"
        )

        model_info = QLabel(
            f"模型：bests.rknn\n输入：{IMG_SIZE} × {IMG_SIZE}\n"
            f"置信度阈值：{CONF_THRESH:.2f}\nNMS 阈值：{IOU_THRESH:.2f}"
        )
        model_info.setObjectName("MutedLabel")
        model_info.setWordWrap(True)

        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(14)
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.alert_label)
        info_layout.addSpacing(8)
        info_layout.addWidget(class_title)
        info_layout.addWidget(class_info)
        info_layout.addWidget(model_info)
        info_layout.addStretch(1)

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(video_card, 7)
        content.addWidget(info_card, 3)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(16)
        root.addLayout(header)
        root.addLayout(content, 1)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.video_label.setText("摄像头读取失败")
            return

        input_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.expand_dims(input_img, 0)

        outputs = self.rknn.inference(inputs=[input_img])
        input_data = [
            np.transpose(outputs[0].reshape(3, -1, *outputs[0].shape[-2:]), (2, 3, 0, 1)),
            np.transpose(outputs[1].reshape(3, -1, *outputs[1].shape[-2:]), (2, 3, 0, 1)),
            np.transpose(outputs[2].reshape(3, -1, *outputs[2].shape[-2:]), (2, 3, 0, 1)),
        ]

        boxes, classes, scores = yolov5_post_process(input_data, CONF_THRESH, IOU_THRESH)
        detected = []

        if boxes is not None:
            scale_x = frame.shape[1] / IMG_SIZE
            scale_y = frame.shape[0] / IMG_SIZE
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                name = CLASS_NAMES[int(cls)]
                color = CLASS_COLORS.get(name, (0, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{name} {score:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                if name in ALERT_CLASSES:
                    detected.append(name)

        unique_detected = sorted(set(detected))
        FusionState.detected_behaviors = unique_detected
        FusionState.shared_yolo_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        if unique_detected:
            name_map = {"phone": "使用手机", "smoke": "吸烟"}
            display = "、".join(name_map.get(name, name) for name in unique_detected)
            self.last_alert_text = f"⚠ 检测到危险行为\n{display}"
            self.alert_label.setText(self.last_alert_text)
            self.alert_label.setStyleSheet(
                f"color: {DANGER}; background: {PLOT}; border: 1px solid {DANGER}; "
                "border-radius: 14px; padding: 16px; font-size: 21px; font-weight: 700;"
            )
        else:
            self.last_alert_text = "当前未检测到危险行为"
            self.alert_label.setText(self.last_alert_text)
            self.alert_label.setStyleSheet(
                f"color: {SUCCESS}; background: {PLOT}; border: 1px solid {SUCCESS}; "
                "border-radius: 14px; padding: 16px; font-size: 21px; font-weight: 700;"
            )

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.rknn.release()
        event.accept()
