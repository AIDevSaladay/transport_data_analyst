import cv2
import numpy as np
from ultralytics import YOLO
import torch
from byte_tracker import BYTETracker  # импорт ByteTrack
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.types as T
from datetime import datetime
import kafka
import boto3
import os

# ========== Конфигурация ==========
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'raw_detections'
MINIO_ENDPOINT = 'minio:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET = 'bronze-video'
CLICKHOUSE_URL = 'jdbc:clickhouse://clickhouse:8123/default'
CAMERA_ID = 'CAM-001'  # можно брать из названия потока
VIDEO_SOURCE = 0  # для веб-камеры, или путь к файлу, или RTSP-URL

# ========== Инициализация ==========
# YOLO модель
model = YOLO('yolov8n.pt')  # nano версия для скорости, можно medium если GPU мощный
# Фильтр классов: 2=car, 3=motorcycle, 5=bus, 7=truck (COCO)
VEHICLE_CLASSES = {2, 3, 5, 7}

# Трекер
tracker = BYTETracker(frame_rate=30)

# Kafka producer
producer = kafka.KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# S3 клиент (MinIO)
s3 = boto3.client('s3',
                  endpoint_url=f'http://{MINIO_ENDPOINT}',
                  aws_access_key_id=MINIO_ACCESS_KEY,
                  aws_secret_access_key=MINIO_SECRET_KEY
                  )

# Spark сессия (для агрегации)
spark = SparkSession.builder \
    .appName("StreamingAggregator") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,ru.yandex.clickhouse:clickhouse-jdbc:0.4.0") \
    .getOrCreate()


# ========== Функции обработки кадров ==========
def process_frame(frame, frame_id, timestamp):
    """Детекция + трекинг одного кадра"""
    results = model(frame, classes=list(VEHICLE_CLASSES), conf=0.3)
    detections = []
    if len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            # Конвертируем в формат ByteTrack: [x1, y1, x2, y2, score, class_id]
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                detections.append([x1, y1, x2, y2, conf, cls_id])

    # Трекинг
    tracked_objects = tracker.update(np.array(detections), [frame.shape[0], frame.shape[1]])

    # Формируем события
    events = []
    for obj in tracked_objects:
        event = {
            'detection_id': str(uuid.uuid4()),
            'camera_id': CAMERA_ID,
            'timestamp': timestamp,
            'frame_id': frame_id,
            'track_id': int(obj.track_id),
            'vehicle_type_id': int(obj.class_id),  # можно маппить в свой справочник
            'bbox_x1': float(obj.x1),
            'bbox_y1': float(obj.y1),
            'bbox_x2': float(obj.x2),
            'bbox_y2': float(obj.y2),
            'confidence': float(obj.score),
            'speed_kmh': 0.0  # будет заполняться позже при наличии калибровки
        }
        events.append(event)
        # Отправляем в Kafka
        producer.send(KAFKA_TOPIC, value=event)
    return events


# ========== Основной цикл захвата видео ==========
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Подготовка записи видео с разметкой (для сохранения в MinIO)
    out_video_path = f'/tmp/processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = datetime.now().isoformat()

        # Детекция и трекинг
        events = process_frame(frame, frame_id, timestamp)

        # Рисуем bounding box на кадре (для сохранения видео)
        for ev in events:
            x1, y1, x2, y2 = ev['bbox_x1'], ev['bbox_y1'], ev['bbox_x2'], ev['bbox_y2']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{ev['track_id']}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        out_video.write(frame)

        # Каждые 100 кадров загружаем видео в MinIO (или по окончании)
        if frame_id % 100 == 0:
            # Асинхронная загрузка
            s3.upload_file(out_video_path, MINIO_BUCKET,
                           f"camera_{CAMERA_ID}/{datetime.now().strftime('%Y/%m/%d')}/segment_{frame_id}.mp4")

        frame_id += 1

    cap.release()
    out_video.release()
    producer.flush()
    producer.close()


if __name__ == '__main__':
    main()