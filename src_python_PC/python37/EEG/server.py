# # server.py
# import asyncio
# import websockets
# import json
# import threading
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# import time
#
# # 注释掉 LSL 相关部分
# # from pylsl import StreamInlet, resolve_byprop
# # print("Looking for EEG stream...")
# # streams = resolve_byprop('type', 'EEG', timeout=5)
# # inlet = StreamInlet(streams[0])
#
# clients = set()
#
# # 模拟信号参数
# fs = 250
# t = 0
# freqs = [1, 2, 3, 4, 5, 6, 7, 8]  # 每通道不同频率
#
# buffer_size = fs
# eeg_buffer = np.zeros((8, buffer_size))
#
# # 模拟正弦信号替代EEG输入
# def generate_sine_wave():
#     global t
#     data = [np.sin(2 * np.pi * f * t) * 10000 for f in freqs]
#     t += 1/fs
#     return data
#
# async def eeg_producer():
#     while True:
#         data = generate_sine_wave()
#         eeg_buffer[:, :-1] = eeg_buffer[:, 1:]
#         eeg_buffer[:, -1] = data
#         # 广播给所有客户端
#         for client in list(clients):
#             try:
#                 await client.send(json.dumps(data))
#             except:
#                 clients.remove(client)
#         await asyncio.sleep(1/fs)
#
# async def handler(websocket, path):
#     print("客户端已连接")
#     clients.add(websocket)
#     try:
#         await websocket.wait_closed()
#     finally:
#         clients.remove(websocket)
#         print("客户端已断开")
#
# def start_websocket_server():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     start_server = websockets.serve(handler, "0.0.0.0", 8765)
#     loop.run_until_complete(start_server)
#     loop.create_task(eeg_producer())
#     loop.run_forever()
#
# # PC本地绘图
# def animate(i):
#     for ch in range(8):
#         lines[ch].set_ydata(eeg_buffer[ch])
#     return lines
#
# fig, ax = plt.subplots(8, 1, figsize=(10, 8))
# lines = []
# for i in range(8):
#     l, = ax[i].plot(np.zeros(buffer_size))
#     ax[i].set_ylim(-12000, 12000)
#     ax[i].set_ylabel(f"Ch {i+1}")
#     lines.append(l)
# ani = animation.FuncAnimation(fig, animate, interval=40)
#
# # 启动线程+GUI
# threading.Thread(target=start_websocket_server, daemon=True).start()
# plt.show()




# # server.py：真实脑电数据源版本
# import asyncio
# import websockets
# import json
# import threading
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# import time
# from pylsl import StreamInlet, resolve_byprop
#
# clients = set()
#
# # EEG参数
# fs = 250
# buffer_size = fs
# eeg_buffer = np.zeros((8, buffer_size))
#
# # =============== LSL 设置 ================
# print("Looking for EEG stream...")
# streams = resolve_byprop('type', 'EEG', timeout=10)
# if len(streams) == 0:
#     raise RuntimeError("没有找到类型为 'EEG' 的 LSL 流，请确认 OpenBCI_GUI 正在运行并打开 LSL 流")
# inlet = StreamInlet(streams[0])
# print("EEG stream found:", inlet.info().name())
#
# # ============ 异步发送 EEG 数据 ==============
# async def eeg_producer():
#     while True:
#         chunk, timestamps = inlet.pull_chunk(timeout=1.0)
#         if chunk:
#             for sample in chunk:
#                 if len(sample) >= 8:
#                     eeg_buffer[:, :-1] = eeg_buffer[:, 1:]
#                     eeg_buffer[:, -1] = sample[:8]
#                     # 发送最新一帧
#                     for client in list(clients):
#                         try:
#                             await client.send(json.dumps(sample[:8]))
#                         except:
#                             clients.remove(client)
#         await asyncio.sleep(0)  # 立即让出事件循环
#
# # WebSocket 处理
# async def handler(websocket, path):
#     print("客户端已连接")
#     clients.add(websocket)
#     try:
#         await websocket.wait_closed()
#     finally:
#         clients.remove(websocket)
#         print("客户端已断开")
#
# def start_websocket_server():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     start_server = websockets.serve(handler, "0.0.0.0", 8765)
#     loop.run_until_complete(start_server)
#     loop.create_task(eeg_producer())
#     loop.run_forever()
#
# # ============== 本地绘图 ================
# def animate(i):
#     for ch in range(8):
#         lines[ch].set_ydata(eeg_buffer[ch])
#     return lines
#
# fig, ax = plt.subplots(8, 1, figsize=(12, 10))  # 增大图形尺寸
# lines = []
#
# for i in range(8):
#     l, = ax[i].plot(np.zeros(buffer_size))
#     ax[i].set_ylim(-50000, 50000)  # 你可以根据实际EEG幅度调整
#     ax[i].set_ylabel(f"Ch {i+1}")
#     lines.append(l)
#
# ani = animation.FuncAnimation(fig, animate, interval=40, cache_frame_data=False)
#
# # 启动后台WebSocket服务 + 显示图形
# threading.Thread(target=start_websocket_server, daemon=True).start()
# plt.tight_layout()
# plt.show()






# server.py：真实脑电数据源版本
import asyncio
import websockets
import json
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop

clients = set()

# EEG参数
fs = 250
buffer_size = 1000  # 增大缓冲区以显示更长时间的数据
eeg_buffer = np.zeros((8, buffer_size))

# =============== LSL 设置 ================
print("Looking for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=10)
if len(streams) == 0:
    raise RuntimeError("没有找到类型为 'EEG' 的 LSL 流，请确认 OpenBCI_GUI 正在运行并打开 LSL 流")
inlet = StreamInlet(streams[0])
print("EEG stream found:", inlet.info().name())

# ============ 异步发送 EEG 数据 ==============
async def eeg_producer():
    while True:
        chunk, timestamps = inlet.pull_chunk(timeout=1.0)
        if chunk:
            for sample in chunk:
                if len(sample) >= 8:
                    eeg_buffer[:, :-1] = eeg_buffer[:, 1:]
                    eeg_buffer[:, -1] = sample[:8]
                    # 发送最新一帧
                    for client in list(clients):
                        try:
                            await client.send(json.dumps(sample[:8]))
                        except:
                            clients.remove(client)
        await asyncio.sleep(0)  # 立即让出事件循环

# WebSocket 处理
async def handler(websocket, path):
    print("客户端已连接")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print("客户端已断开")

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(handler, "0.0.0.0", 8765)
    loop.run_until_complete(start_server)
    loop.create_task(eeg_producer())
    loop.run_forever()

# ============== 本地绘图 ================
def animate(i):
    for ch in range(8):
        lines[ch].set_ydata(eeg_buffer[ch])
    return lines

# 创建图形和坐标轴
fig, ax = plt.subplots(8, 1, figsize=(10, 8))  # 增大图形尺寸
lines = []

# 优化幅度范围和显示设置
y_min = -1000  # 根据数据范围调整
y_max = 1000

for i in range(8):
    line, = ax[i].plot(np.zeros(buffer_size))
    lines.append(line)
    ax[i].set_ylim(y_min, y_max)  # 优化幅度范围
    ax[i].set_ylabel(f"Ch {i + 1}", fontsize=9)  # 减小字体避免重叠
    ax[i].tick_params(axis='both', labelsize=8)  # 减小刻度字体

# 增加标题和共享X轴
fig.suptitle('EEG Real-time Monitoring (Server View)', fontsize=12)
plt.subplots_adjust(hspace=0.4)  # 增加子图间距

# 降低刷新频率以减少CPU负载
ani = animation.FuncAnimation(
    fig, animate,
    interval=40,  # 从40ms增加到100ms
    cache_frame_data=False
)

# 启动后台WebSocket服务 + 显示图形
threading.Thread(target=start_websocket_server, daemon=True).start()
plt.tight_layout()
plt.show()