#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQTT 桥接脚本：订阅 MQTT 主题，将传感器数据转为标准化格式后 POST 到 ingest API。
用于快速验证多源数据接入，无需修改后端。

依赖：pip install paho-mqtt requests

用法：
  python mqtt_bridge.py
  python mqtt_bridge.py --broker localhost --port 1883 --topic sensors/#
  python mqtt_bridge.py --simulate   # 无 MQTT 时模拟发布并自消费
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, Optional

try:
    import paho.mqtt.client as mqtt
    import requests
except ImportError as e:
    print("请安装依赖: pip install paho-mqtt requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_BROKER = "localhost"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "sensors/#"
DEFAULT_INGEST_URL = "http://localhost:4999/api/data/ingest"


def to_ingest_record(payload: Any) -> Optional[Dict[str, Any]]:
    """
    将 MQTT 消息体转为 ingest 可接受的格式。
    支持：
    - 已是标准化格式：{ "ts": 1730350800, "data": { "crack_1": 0.52, ... } }
    - 已是扁平格式：{ "timestamp": 1730350800, "crack_1": 0.52, ... }
    - 纯数据：{ "crack_1": 0.52, ... } → 自动加 ts
    """
    if not isinstance(payload, dict):
        return None
    if "ts" in payload or "timestamp" in payload or "time" in payload:
        return payload
    ts = int(time.time())
    return {"ts": ts, "data": payload}


def post_ingest(record: Dict[str, Any], url: str) -> bool:
    """POST 单条记录到 ingest API"""
    try:
        r = requests.post(url, json=record, timeout=5)
        if r.status_code == 200:
            return True
        logger.warning("ingest 返回 %s: %s", r.status_code, r.text[:200])
        return False
    except Exception as e:
        logger.error("ingest 请求失败: %s", e)
        return False


def on_mqtt_message(client, userdata, msg):
    """MQTT 消息回调"""
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("无法解析 JSON [%s]: %s", topic, e)
        return

    record = to_ingest_record(payload)
    if not record:
        logger.warning("跳过无效消息 [%s]", topic)
        return

    url = userdata.get("ingest_url", DEFAULT_INGEST_URL)
    if post_ingest(record, url):
        logger.info("已上报 [%s] -> ingest (buffer 见 API 响应)", topic)
    else:
        logger.warning("上报失败 [%s]", topic)


def run_mqtt_bridge(broker: str, port: int, topic: str, ingest_url: str):
    """运行 MQTT 桥接"""
    client = mqtt.Client(client_id="predictive-shm-bridge")
    client.user_data_set({"ingest_url": ingest_url})
    client.on_message = on_mqtt_message

    try:
        client.connect(broker, port, 60)
        client.subscribe(topic)
        logger.info("已连接 %s:%d，订阅 %s，ingest=%s", broker, port, topic, ingest_url)
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        client.disconnect()


def run_simulate(ingest_url: str, interval: float = 10.0):
    """
    模拟模式：无 MQTT broker 时，本地生成数据并 POST 到 ingest。
    用于快速验证 ingest 与前端展示。
    """
    logger.info("模拟模式：每 %.1f 秒上报一条，ingest=%s", interval, ingest_url)
    i = 0
    try:
        while True:
            i += 1
            ts = int(time.time())
            crack = 0.1 + 0.05 * (i % 20) + 0.02 * (i % 7)
            tilt_x = 0.01 * (i % 15) + 0.005 * (i % 5)
            record = {
                "ts": ts,
                "data": {
                    "crack_1": round(crack, 4),
                    "crack_2": round(crack + 0.01, 4),
                    "crack_3": round(crack + 0.02, 4),
                    "tilt_x_1": round(tilt_x, 4),
                    "tilt_x_2": round(tilt_x + 0.002, 4),
                    "tilt_x_3": round(tilt_x - 0.001, 4),
                    "tilt_x_4": round(tilt_x + 0.001, 4),
                    "tilt_y_1": 0.01, "tilt_y_2": 0.012, "tilt_y_3": 0.011, "tilt_y_4": 0.013,
                    "settlement_1": 0.5, "settlement_2": 0.52, "settlement_3": 0.51, "settlement_4": 0.49,
                    "water_level": 50.0, "temperature": 20.0,
                },
            }
            if post_ingest(record, ingest_url):
                logger.info("模拟上报 #%d 成功", i)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("用户中断")


def main():
    parser = argparse.ArgumentParser(description="MQTT 桥接：订阅 MQTT → POST ingest")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker 地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT 端口")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="订阅主题，支持 # 通配")
    parser.add_argument("--ingest", default=DEFAULT_INGEST_URL, help="ingest API 地址")
    parser.add_argument("--simulate", action="store_true", help="模拟模式（无 MQTT 时使用）")
    parser.add_argument("--interval", type=float, default=10.0, help="模拟模式上报间隔（秒）")
    args = parser.parse_args()

    if args.simulate:
        run_simulate(args.ingest, args.interval)
    else:
        run_mqtt_bridge(args.broker, args.port, args.topic, args.ingest)


if __name__ == "__main__":
    main()
