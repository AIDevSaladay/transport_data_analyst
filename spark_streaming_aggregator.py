import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import requests
import json

CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 8123  # стандартный HTTP-порт ClickHouse
CLICKHOUSE_DB = 'default'
CLICKHOUSE_TABLE = 'fact_camera_detections_realtime'
CLICKHOUSE_URL = f'http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASSWORD = 'user'

spark = SparkSession.builder \
    .appName("TrafficStreamingAggregator") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

# Схема данных из Kafka
schema = StructType([
    StructField("detection_id", StringType()),
    StructField("camera_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("frame_id", IntegerType()),
    StructField("track_id", IntegerType()),
    StructField("vehicle_type_id", IntegerType()),
    StructField("bbox_x1", DoubleType()),
    StructField("bbox_y1", DoubleType()),
    StructField("bbox_x2", DoubleType()),
    StructField("bbox_y2", DoubleType()),
    StructField("confidence", DoubleType()),
    StructField("speed_kmh", DoubleType())
])

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw_detections") \
    .option("startingOffsets", "latest") \
    .load()

parsed_stream = raw_stream \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*") \
    .withColumn("event_time", to_timestamp("timestamp"))

with_watermark = parsed_stream.withWatermark("event_time", "30 seconds")

windowed_counts = with_watermark \
    .groupBy(
        col("camera_id"),
        window(col("event_time"), "1 minute", "1 minute"),
        col("vehicle_type_id")
    ) \
    .agg(
        count("*").alias("vehicle_count"),
        approx_count_distinct("track_id").alias("unique_vehicles"),  # замена countDistinct
        avg("speed_kmh").alias("avg_speed_kmh")
    )

def write_batch_to_clickhouse(df, epoch_id):
    if df.isEmpty():
        return
    df = df.withColumn("window_start", col("window.start")) \
           .withColumn("window_end", col("window.end")) \
           .drop("window")
    rows = df.toJSON().collect()
    if not rows:
        return
    data = "\n".join(rows)
    params = {
        'user': CLICKHOUSE_USER,
        'password': CLICKHOUSE_PASSWORD,
        'database': CLICKHOUSE_DB,
        'query': f"INSERT INTO {CLICKHOUSE_TABLE} FORMAT JSONEachRow"
    }
    response = requests.post(CLICKHOUSE_URL, params=params, data=data.encode('utf-8'),
                             headers={'Content-Type': 'application/octet-stream'})
    if response.status_code != 200:
        raise Exception(f"ClickHouse insert failed: {response.text}")

query = windowed_counts.writeStream \
    .foreachBatch(write_batch_to_clickhouse) \
    .outputMode("update") \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()