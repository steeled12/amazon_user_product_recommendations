#!/usr/bin/env python3
"""
Collaborative Filtering Module

Implements ALS collaborative filtering using enhanced ratings from sentiment analysis.
"""

from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

class CollaborativeFilter:
    """Handles collaborative filtering using ALS"""
    
    def __init__(self, sc):
        self.sc = sc
        self.cf_model = None
        self.user_id_mapping = {}
        self.product_id_mapping = {}
    
    def train_collaborative_filtering(self, df, user_id_mapping, product_id_mapping):
        """Train collaborative filtering model with hyperparameter tuning using ML ALS"""
        print("\n=== STEP 5: Collaborative Filtering with Hyperparameter Tuning ===")
        
        #df=self.filter_sparse_data(df)

        self.user_id_mapping = user_id_mapping
        self.product_id_mapping = product_id_mapping
        
        # Convert to DataFrame format for ML ALS
        user_mapping = self.user_id_mapping
        product_mapping = self.product_id_mapping
        
        def create_rating_row(row):
            user_idx = user_mapping.get(row.user_id)
            product_idx = product_mapping.get(row.parent_asin)
            if user_idx is not None and product_idx is not None:
                return (user_idx, product_idx, float(row.enhanced_rating))
            return None
        
        # Create DataFrame
        ratings_data = df.rdd.map(create_rating_row).filter(lambda x: x is not None)
        ratings_df = ratings_data.toDF(["user", "item", "rating"])
        ratings_df.cache()
        
        total_ratings = ratings_df.count()
        print(f"   Rating DataFrame created: {total_ratings} ratings")
        
        # Split data for training and validation
        training_df, test_df = ratings_df.randomSplit([0.8, 0.2])
        training_df.cache()
        test_df.cache()
        
        training_count = training_df.count()
        test_count = test_df.count()
        
        print(f"   Training samples: {training_count}")
        print(f"   Test samples: {test_count}")
        
        # Setup ALS model for hyperparameter tuning
        als = ALS(userCol="user", itemCol="item", ratingCol="rating", 
                  coldStartStrategy="drop", nonnegative=True)
        
        # Create parameter grid
        """
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [5, 8]) \
            .addGrid(als.maxIter, [20, 30]) \
            .addGrid(als.regParam, [0.1, 0.2]) \
            .build()
        """
        
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [5]) \
            .addGrid(als.maxIter, [30]) \
            .addGrid(als.regParam, [0.2]) \
            .build()
        
        # Setup evaluator (RMSE)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        
        # Setup TrainValidationSplit
        tvs = TrainValidationSplit(estimator=als,
                                   estimatorParamMaps=param_grid,
                                   evaluator=evaluator,
                                   parallelism=4)
        
        print("   Performing hyperparameter tuning...")
        print(f"   Testing {len(param_grid)} parameter combinations...")
        
        # Train and find best model
        tvs_model = tvs.fit(training_df)
        self.cf_model = tvs_model.bestModel
        
        # Get best parameters
        best_rank = self.cf_model.rank
        best_maxIter = self.cf_model._java_obj.parent().getMaxIter()
        best_regParam = self.cf_model._java_obj.parent().getRegParam()
        
        # Evaluate best model
        test_predictions = self.cf_model.transform(test_df)
        rmse = evaluator.evaluate(test_predictions)
        
        print(f"Best model found:")
        print(f"   Best rank: {best_rank}")
        print(f"   Best maxIter: {best_maxIter}")
        print(f"   Best regParam: {best_regParam:.3f}")
        print(f"   Test RMSE: {rmse:.4f}")
        
        # Store best parameters for reference
        self.best_params = {
            'rank': best_rank,
            'maxIter': best_maxIter,
            'regParam': best_regParam,
            'rmse': rmse
        }
        
        print(f"Collaborative filtering model trained with optimized parameters")
        
        return self.cf_model
    
    def get_recommendations(self, user_id, df, num_recommendations=5):
        """Get recommendations for a user using ML ALS model"""
        if user_id not in self.user_id_mapping:
            print(f"User {user_id} not found in dataset")
            return []
        
        user_idx = self.user_id_mapping[user_id]
        
        # Get user's rated products
        user_products = df.filter(col("user_id") == user_id).select("parent_asin").rdd.map(lambda r: r.parent_asin).collect()
        rated_product_indices = set([self.product_id_mapping[p] for p in user_products if p in self.product_id_mapping])
        
        # Get candidate products (not rated by user)
        candidate_products = [prod_idx for prod_idx in range(len(self.product_id_mapping)) 
                             if prod_idx not in rated_product_indices]
        
        if not candidate_products:
            return []
        
        # Limit candidates for performance
        candidate_products = candidate_products[:100]
        
        # Create DataFrame for predictions
        candidate_df = self.sc.parallelize(
            [(user_idx, prod_idx) for prod_idx in candidate_products]
        ).toDF(["user", "item"])
        
        # Get predictions using ML ALS model
        predictions_df = self.cf_model.transform(candidate_df)
        
        # Clip predictions to valid rating range (1.0 to 5.0)
        from pyspark.sql.functions import when, col as spark_col
        predictions_df = predictions_df.withColumn(
            "prediction_clipped",
            when(spark_col("prediction") < 1.0, 1.0)
            .when(spark_col("prediction") > 5.0, 5.0)
            .otherwise(spark_col("prediction"))
        )
        
        # Collect and sort predictions using clipped values
        predictions = predictions_df.select("item", "prediction_clipped").collect()
        predictions.sort(key=lambda x: x.prediction_clipped, reverse=True)
        top_predictions = predictions[:num_recommendations]
        
        # Convert back to product IDs and enrich with additional data
        idx_to_product = {idx: product for product, idx in self.product_id_mapping.items()}
        recommendations = []
        
        for pred in top_predictions:
            product_id = idx_to_product[pred.item]
            
            # Get product information from the dataframe
            product_info = df.filter(col("parent_asin") == product_id).select(
                "sentiment_score", "title"
            ).first()
            
            recommendations.append({
                'product_id': product_id,
                'final_rating': pred.prediction_clipped,
                'sentiment_score': product_info.sentiment_score if product_info and product_info.sentiment_score else 0.0,
                'product_title': product_info.title if product_info and product_info.title else 'Unknown Product'
            })
        
        return recommendations
    
    """     
    def filter_sparse_data(self, df):
        # Filter out users and items with too few ratings
        print("\n=== Filtering Sparse Data ===")
        
        # Remove users with fewer than 3 ratings
        user_counts = df.groupBy("user_id").count()
        active_users = user_counts.filter(col("count") >= 3)
        df_filtered = df.join(active_users.select("user_id"), "user_id")
        
        # Remove items with fewer than 3 ratings
        item_counts = df_filtered.groupBy("parent_asin").count()
        popular_items = item_counts.filter(col("count") >= 3)
        df_filtered = df_filtered.join(popular_items.select("parent_asin"), "parent_asin")
        
        print(f"   Original ratings: {df.count()}")
        print(f"   Filtered ratings: {df_filtered.count()}")
        
        return df_filtered 
    """
    

    def get_model_info(self):
        """Get collaborative filtering model information"""
        if self.cf_model is None:
            return None
        
        info = {
            'model_type': 'ALS',
            'num_users': len(self.user_id_mapping),
            'num_products': len(self.product_id_mapping),
            'model_trained': True
        }
        
        # Add best parameters if available
        if hasattr(self, 'best_params'):
            info.update(self.best_params)
        
        return info
