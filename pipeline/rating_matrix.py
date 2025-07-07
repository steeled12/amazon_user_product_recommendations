#!/usr/bin/env python3
"""
Rating Matrix Module

Creates enhanced rating matrix by combining original ratings with sentiment scores.
"""

from pyspark.sql.functions import col

class RatingMatrixCreator:
    """Creates sentiment-enhanced rating matrix"""
    
    def __init__(self):
        self.user_id_mapping = {}
        self.product_id_mapping = {}
    
    def create_rating_matrix(self, df):
        """Create rating matrix from sentiment analysis"""
        print("\n=== STEP 4: Creating Rating Matrix from Sentiment ===")
        
        # Extract sentiment-enhanced ratings
        # Use prediction probability as sentiment score and combine with original rating
        # First, extract the probability array from the probability vector
        from pyspark.ml.linalg import VectorUDT
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        
        # UDF to extract probability from probability vector
        def extract_probability(prob_vector):
            """Extract positive probability from probability vector"""
            if prob_vector is not None and len(prob_vector) > 1:
                return float(prob_vector[1])  # Probability of positive class
            return 0.5  # Default neutral probability
        
        extract_prob_udf = udf(extract_probability, DoubleType())
        
        df = df.withColumn(
            "sentiment_score", 
            extract_prob_udf(col("probability"))
        )
        
        # Create enhanced rating: combine original rating with sentiment
        df = df.withColumn(
            "enhanced_rating",
            col("rating") * 0.7 + col("sentiment_score") * 5 * 0.3
        )
        
        # Create user and product mappings for ALS
        users = df.select("user_id").distinct().collect()
        products = df.select("parent_asin").distinct().collect()
        
        self.user_id_mapping = {row.user_id: idx for idx, row in enumerate(users)}
        self.product_id_mapping = {row.parent_asin: idx for idx, row in enumerate(products)}
        
        print(f" Rating matrix created:")
        print(f"   Users: {len(self.user_id_mapping)}")
        print(f"   Products: {len(self.product_id_mapping)}")
        
        # Show sample of enhanced ratings
        sample_ratings = df.select("rating", "sentiment_score", "enhanced_rating").limit(5).collect()
        print("   Sample enhanced ratings:")
        for row in sample_ratings:
            print(f"     Original: {row.rating:.1f}, Sentiment: {row.sentiment_score:.3f}, Enhanced: {row.enhanced_rating:.2f}")
        
        return df
    
    def get_mappings(self):
        """Get user and product ID mappings"""
        return self.user_id_mapping, self.product_id_mapping
    
    def get_matrix_stats(self):
        """Get rating matrix statistics"""
        return {
            'num_users': len(self.user_id_mapping),
            'num_products': len(self.product_id_mapping),
            'matrix_density': len(self.user_id_mapping) * len(self.product_id_mapping)
        }
