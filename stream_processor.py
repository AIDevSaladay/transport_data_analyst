import cv2
import numpy as np
from ultralytics import YOLO
import json
import uuid
from datetime import datetime
import kafka
import boto3
import argparse

# ========== Конфигурация ==========
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'raw_detections'
MINIO_ENDPOINT = 'localhost:9001'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET = 'bronze-video'
CAMERA_ID = 'CAM-001'
VIDEO_SOURCE = './video/istockphoto-1066941272-640_adpp_is.mp4'  # для веб-камеры, или путь к файлу, или RTSP-URL

# ========== Инициализация ==========
model = YOLO('yolov8n.pt')
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Kafka producer./video/istockphoto-1066941272-640_adpp_is.mp4
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

def process_frame(frame, frame_id, timestamp):
    """Детекция + трекинг (встроенный YOLO) и отправка в Kafka"""
    # Запускаем трекинг на кадре
    results = model.track(frame, classes=list(VEHICLE_CLASSES), conf=0.3, persist=True, tracker="bytetrack.yaml")
    
    events = []
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            track_id = int(box.id[0].item())
            
            event = {
                'detection_id': str(uuid.uuid4()),
                'camera_id': CAMERA_ID,
                'timestamp': timestamp,
                'frame_id': frame_id,
                'track_id': track_id,
                'vehicle_type_id': cls_id,
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2,
                'confidence': conf,
                'speed_kmh': 0.0
            }
            events.append(event)
            # Вместо producer.send(KAFKA_TOPIC, value=event)
            future = producer.send(KAFKA_TOPIC, value=event)
            try:
                record_metadata = future.get(timeout=10)
                print(f"Sent {event['detection_id']} to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")
            except Exception as e:
                print(f"Failed to send: {e}")
    return events

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеопоток")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Подготовка записи видео с разметкой
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_path = f'/tmp/processed_{timestamp_str}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = datetime.now().isoformat()
            events = process_frame(frame, frame_id, timestamp)
            
            # Рисуем bounding box на кадре
            for ev in events:
                x1, y1, x2, y2 = ev['bbox_x1'], ev['bbox_y1'], ev['bbox_x2'], ev['bbox_y2']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"ID:{ev['track_id']}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            out_video.write(frame)
            
            # Каждые 100 кадров загружаем видео в MinIO
            if frame_id % 100 == 0:
                s3.upload_file(out_video_path, MINIO_BUCKET,
                               f"camera_{CAMERA_ID}/{datetime.now().strftime('%Y/%m/%d')}/segment_{frame_id}.mp4")
            
            frame_id += 1
    finally:
        cap.release()
        out_video.release()
        producer.flush()
        producer.close()
        print("Обработка завершена, видео сохранено в", out_video_path)

if __name__ == '__main__':
    main()