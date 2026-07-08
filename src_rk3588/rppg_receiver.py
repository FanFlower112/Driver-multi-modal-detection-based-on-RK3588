"""RK3588 端 rPPG WebSocket 接收线程。

电脑端接口：ws://<电脑局域网IP>:8000/rppg_ws
收到 JSON 后通过 Qt 信号交给 RPPGWidget.update_rppg_data()。
"""

import asyncio
import json

import websockets
from PyQt5.QtCore import QThread, pyqtSignal


class RPPGReceiver(QThread):
    data_received = pyqtSignal(dict)
    connection_changed = pyqtSignal(str, bool)

    def __init__(self, uri, parent=None, reconnect_seconds=2.0):
        super().__init__(parent)
        self.uri = uri
        self.reconnect_seconds = float(reconnect_seconds)
        self._running = True

    def run(self):
        try:
            asyncio.run(self._receive_forever())
        except Exception as exc:
            self.connection_changed.emit(f"接收线程异常：{exc}", False)

    async def _receive_forever(self):
        while self._running:
            try:
                self.connection_changed.emit(
                    f"正在连接电脑端：{self.uri}", False
                )

                async with websockets.connect(
                    self.uri,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=3,
                    max_size=4 * 1024 * 1024,
                ) as websocket:
                    self.connection_changed.emit("rPPG 数据通道已连接", True)

                    async for raw_message in websocket:
                        if not self._running:
                            break

                        try:
                            payload = json.loads(raw_message)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(payload, dict):
                            continue

                        message_type = payload.get("type")
                        if message_type not in (None, "rppg"):
                            continue

                        self.data_received.emit(payload)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    self.connection_changed.emit(
                        f"rPPG 连接断开，正在重连：{exc}", False
                    )
                    await asyncio.sleep(self.reconnect_seconds)

    def stop(self):
        self._running = False
        self.requestInterruption()
        self.wait(3000)
