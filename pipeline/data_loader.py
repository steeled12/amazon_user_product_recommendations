#!/usr/bin/env python3
"""
Data Loading and Exploration Module

Handles Amazon dataset loading and basic exploratory analysis with visualizations.
"""

from datasets import load_dataset

class DataLoader:
    """Handles dataset loading and exploration with visualizations"""
    
    def __init__(self, spark):
        self.spark = spark
        self.df = None
    
    def load_and_explore_dataset(self, sample_size=10000):
        """Load and explore Amazon dataset"""
        print(f"\n=== STEP 2: Loading and Exploring Dataset ({sample_size} samples) ===")
        
        # Load Amazon dataset
        print("Loading Amazon Reviews 2023 dataset...")
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                              "raw_review_Appliances", 
                              split=f"full[:{sample_size}]",
                              trust_remote_code=True)
        
        metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                                "raw_meta_Appliances", 
                                split="full",
                                trust_remote_code=True)
        
        # Save to parquet files
        dataset.to_parquet("temp_reviews.parquet")
        metadata.to_parquet("temp_metadata.parquet")
        
        # Load parquet files as Spark DataFrames
        reviews_df = self.spark.read.parquet("temp_reviews.parquet")
        metadata_df = self.spark.read.parquet("temp_metadata.parquet")
        
        # Select only essential columns and filter nulls
        essential_columns = ['user_id', 'parent_asin', 'rating', 'text']
        reviews_df = reviews_df.select(*essential_columns).dropna()
        metadata_df = metadata_df.select('parent_asin', 'title').dropna()
        
        # Merge with metadata to get product titles
        self.df = reviews_df.join(metadata_df, on='parent_asin', how='left')

        # Exploratory analysis
        total_reviews = self.df.count()
        unique_users = self.df.select('user_id').distinct().count()
        unique_products = self.df.select('parent_asin').distinct().count()
        
        print(f" Dataset loaded: {total_reviews} reviews")
        print("\n Dataset Overview:")
        print(f"   Users: {unique_users}")
        print(f"   Products: {unique_products}")
        
        # Rating distribution
        rating_dist = self.df.groupBy('rating').count().orderBy('rating').collect()
        print("   Rating Distribution:")
        for row in rating_dist:
            print(f"     {row['rating']} stars: {row['count']} reviews")
        
        return self.df
    
    def get_dataset_stats(self):
        """Get basic dataset statistics"""
        if self.df is None:
            return None
        
        return {
            'total_reviews': self.df.count(),
            'unique_users': self.df.select('user_id').distinct().count(),
            'unique_products': self.df.select('parent_asin').distinct().count(),
            'avg_rating': self.df.agg({'rating': 'avg'}).collect()[0][0]
        }
