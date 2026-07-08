"""RK3588 驾驶员状态监测系统统一暗色主题。"""

BACKGROUND = "#0b1220"
PANEL = "#111827"
CARD = "#151e2e"
PLOT = "#020617"
BORDER = "#273449"
TEXT = "#e5edf8"
MUTED = "#8fa3bd"
ACCENT = "#38bdf8"
ACCENT_DARK = "#0c4a6e"
SUCCESS = "#34d399"
WARNING = "#f59e0b"
DANGER = "#fb7185"

APP_STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
}}

QFrame#Sidebar {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 18px;
}}

QFrame#Card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
}}

QFrame#MetricCard {{
    background-color: {PLOT};
    border: 1px solid {BORDER};
    border-radius: 14px;
}}

QLabel#PageTitle {{
    color: {TEXT};
    font-size: 28px;
    font-weight: 700;
}}

QLabel#PageSubtitle {{
    color: {MUTED};
    font-size: 15px;
}}

QLabel#MetricTitle {{
    color: {MUTED};
    font-size: 14px;
}}

QLabel#MetricValue {{
    color: {TEXT};
    font-size: 28px;
    font-weight: 700;
}}

QLabel#SectionTitle {{
    color: {TEXT};
    font-size: 19px;
    font-weight: 700;
}}

QLabel#MutedLabel {{
    color: {MUTED};
    font-size: 14px;
}}

QPushButton#NavButton {{
    min-height: 54px;
    padding: 8px 18px;
    margin: 5px 10px;
    text-align: left;
    color: #b9c7d9;
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 12px;
    font-size: 19px;
    font-weight: 600;
}}

QPushButton#NavButton:hover {{
    color: #ffffff;
    background-color: #1b283b;
    border-color: #32445f;
}}

QPushButton#NavButton:checked {{
    color: #ffffff;
    background-color: #123b59;
    border-color: {ACCENT};
}}

QTextEdit {{
    color: #dbe7f5;
    background-color: {PLOT};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 10px;
    selection-background-color: {ACCENT_DARK};
}}

QStackedWidget {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 18px;
}}

QToolTip {{
    color: {TEXT};
    background-color: {PANEL};
    border: 1px solid {BORDER};
}}
"""
