#!/usr/bin/env python3
"""
Sentiment Analysis Module

Performs sentiment analysis using PySpark MLlib with ML Pipeline.
"""

import re
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class SentimentAnalyzer:
    """Handles sentiment analysis using MLlib"""
    
    def __init__(self):
        self.sentiment_model = None
        self.last_auc = 0.0
    
    def clean_text(self, text):
        """Clean text for sentiment analysis"""
        if not text:
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        return text.lower().strip()
    
    def perform_sentiment_analysis(self, df):
        """Perform sentiment analysis using MLlib"""
        print("\n=== STEP 3: Sentiment Analysis using MLlib ===")
        
        # Clean text UDF
        clean_text_udf = udf(self.clean_text, StringType())
        
        # Add cleaned text
        df = df.withColumn("cleaned_text", clean_text_udf(col("text")))
        
        # Create sentiment labels from ratings (4-5 = positive, 1-2 = negative)
        df = df.withColumn(
            "sentiment_label", 
            when(col("rating") >= 4, 1.0).otherwise(0.0)
        )
        
        # Filter out neutral ratings for training
        training_df = df.filter((col("rating") <= 2) | (col("rating") >= 4))
        
        print(f"   Training samples: {training_df.count()}")
        
        # Create ML pipeline for sentiment analysis
        tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="tokens")
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="tf_features", numFeatures=1000)
        idf = IDF(inputCol="tf_features", outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="sentiment_label")
        
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
        
        # Split data for training
        train_data, test_data = training_df.randomSplit([0.8, 0.2], seed=42)
        
        # Train sentiment model
        print("   Training sentiment model...")
        self.sentiment_model = pipeline.fit(train_data)
        
        # Evaluate model
        predictions = self.sentiment_model.transform(test_data)
        evaluator = BinaryClassificationEvaluator(labelCol="sentiment_label")
        accuracy = evaluator.evaluate(predictions)
        self.last_auc = accuracy
        
        print(f" Sentiment model trained (AUC: {accuracy:.3f})")
        
        # Apply sentiment analysis to all reviews
        df_with_sentiment = self.sentiment_model.transform(df)
        
        return df_with_sentiment
    
    def get_sentiment_statistics(self, df):
        """Get sentiment analysis statistics"""
        if "sentiment_label" not in df.columns:
            return None
        
        positive_count = df.filter(col("sentiment_label") == 1.0).count()
        negative_count = df.filter(col("sentiment_label") == 0.0).count()
        total_count = df.count()
        
        return {
            'positive_reviews': positive_count,
            'negative_reviews': negative_count,
            'total_reviews': total_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0
        }
