# fusion_state.py

class FusionState:
    latest_eeg_score = 0.0
    latest_eye_score = 0.0
    detected_behaviors = []  # from YOLO
    shared_yolo_frame = None        # YOLO 检测后的图像帧 (BGR格式)
    eeg_pred_class = -1   # 0=清醒, 1=疲劳