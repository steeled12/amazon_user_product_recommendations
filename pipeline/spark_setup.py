#!/usr/bin/env python3
"""
Spark Setup Module

Handles Spark session creation and configuration for the pipeline.
"""

from pyspark.sql import SparkSession
import os

class SparkManager:
    """Manages Spark session creation and configuration"""
    
    def __init__(self):
        self.spark = None
        self.sc = None
    
    def setup_spark(self):
        print("=== STEP 1: Setting up Spark Context ===")

        os.environ['PYSPARK_PYTHON'] = 'python3'
        os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

        self.spark = SparkSession.builder \
            .appName("Amazon_Sentiment_CF_Pipeline") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "3g") \
            .config("spark.executor.cores", "2") \
            .config("spark.default.parallelism", "16") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.driver.extraJavaOptions", "-Xss4m -XX:+UseG1GC") \
            .config("spark.executor.extraJavaOptions", "-Xss4m -XX:+UseG1GC") \
            .getOrCreate()
        
        self.sc = self.spark.sparkContext
        self.sc.setLogLevel("ERROR") 
        
        print(f" Spark Version: {self.spark.version}")
        print(f" Spark Context created successfully")
        
        return self.spark, self.sc
    
    
    def stop_spark(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            print("Spark session stopped.")
        
    
    