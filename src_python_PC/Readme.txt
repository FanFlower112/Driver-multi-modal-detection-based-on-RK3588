脑电：
	lslTest.py和server.py
	lslTest.py：PC端通过lsl流实时获取openBCI的8通道脑电帽的脑电信号。再利用websocket从PC传输到rk3588上。
眼动：
	Eye\Send_only.py和Eye\Eye_tracking_detection1.py




重点是几个.py文件，主要功能是实现从上位机（如脑电帽、眼动仪）的数据获取到PC上，随后通过websocket传输到rk3588上进行处理。