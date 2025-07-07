#!/usr/bin/env python3
"""
Main Pipeline Module

Orchestrates the complete sentiment-aware recommendation pipeline.
"""

import warnings
from pyspark.sql.functions import col

from .spark_setup import SparkManager
from .data_loader import DataLoader
from .sentiment_analysis import SentimentAnalyzer
from .rating_matrix import RatingMatrixCreator
from .collaborative_filtering import CollaborativeFilter
from .visualization import PipelineVisualizer

warnings.filterwarnings('ignore')

class PipelineOrchestrator:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.spark_manager = SparkManager()
        self.spark = None
        self.sc = None
        self.data_loader = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.rating_creator = RatingMatrixCreator()
        self.collaborative_filter = None
        self.visualizer = PipelineVisualizer()
        self.df = None
        self.recommendations_data = {}
    
    def setup_spark(self):
        """Initialize Spark session"""
        self.spark, self.sc = self.spark_manager.setup_spark()
        self.data_loader = DataLoader(self.spark)
        self.collaborative_filter = CollaborativeFilter(self.sc)
        return self
    
    def load_and_explore_dataset(self, sample_size=10000):
        """Load and explore dataset"""
        self.df = self.data_loader.load_and_explore_dataset(sample_size)
        return self
    
    def perform_sentiment_analysis(self):
        """Perform sentiment analysis"""
        self.df = self.sentiment_analyzer.perform_sentiment_analysis(self.df)
        return self
    
    def create_rating_matrix(self):
        """Create sentiment-enhanced rating matrix"""
        self.df = self.rating_creator.create_rating_matrix(self.df)
        return self
    
    def collaborative_filtering(self):
        """Train collaborative filtering model"""
        user_mapping, product_mapping = self.rating_creator.get_mappings()
        self.collaborative_filter.train_collaborative_filtering(
            self.df, user_mapping, product_mapping
        )
        return self
    
    def get_recommendations(self, user_id, num_recommendations=5):
        """Get recommendations for a user"""
        return self.collaborative_filter.get_recommendations(
            user_id, self.df, num_recommendations
        )
    
    def show_recommendations(self, num_users=3, num_recommendations=5):
        print(f"\nRecommendation System ({num_users} users)")
        
        # Get sample users
        all_users = list(self.rating_creator.user_id_mapping.keys())
        # Filter users with at least 3 reviews
        user_review_counts = self.df.rdd.map(lambda row: (row.user_id, 1)) \
                        .reduceByKey(lambda a, b: a + b) \
                        .filter(lambda x: x[1] >= 3)
        
        users_with_min_reviews = user_review_counts.map(lambda x: x[0]).collect()
        sample_users = users_with_min_reviews[:min(num_users, len(users_with_min_reviews))]
        
        self.recommendations_data = {}
        
        for i, user_id in enumerate(sample_users, 1):
            print(f"\n{i}. Recommendations for user: {user_id[:20]}...")
            
            # Show user's rating history
            user_history = self.df.filter(col("user_id") == user_id) \
                                 .select("parent_asin", "rating", "enhanced_rating", "sentiment_score") \
                                 .limit(3).collect()
            
            print("  User's rating history:")
            for row in user_history:
                print(f"    Product: {row.parent_asin[:15]}... Rating: {row.rating} -> Enhanced: {row.enhanced_rating:.2f} (Sentiment: {row.sentiment_score:.3f})")
            
            # Get recommendations
            recommendations = self.get_recommendations(user_id, num_recommendations)
            self.recommendations_data[user_id] = recommendations
            
            print("   Recommendations:")
            for j, rec in enumerate(recommendations, 1):
                print(f"    {j}. Product: {rec['product_id'][:15]}...")
                print(f"       Predicted Rating: {rec['final_rating']:.2f}")
                print()
        
        return self.recommendations_data
    
    def get_pipeline_summary(self):
        """Get summary of pipeline results"""
        if self.df is None:
            return None
        
        dataset_stats = self.data_loader.get_dataset_stats()
        sentiment_stats = self.sentiment_analyzer.get_sentiment_statistics(self.df)
        matrix_stats = self.rating_creator.get_matrix_stats()
        model_info = self.collaborative_filter.get_model_info()
        
        # Calculate additional stats for visualizations
        all_predicted_ratings = []
        total_recommendations = 0
        
        for user_recs in self.recommendations_data.values():
            total_recommendations += len(user_recs)
            for rec in user_recs:
                all_predicted_ratings.append(rec['final_rating'])
        
        avg_predicted_rating = sum(all_predicted_ratings) / len(all_predicted_ratings) if all_predicted_ratings else 0
        
        return {
            'total_reviews': dataset_stats['total_reviews'],
            'unique_users': dataset_stats['unique_users'],
            'unique_products': dataset_stats['unique_products'],
            'positive_reviews': sentiment_stats['positive_reviews'],
            'negative_reviews': sentiment_stats['negative_reviews'],
            'positive_ratio': sentiment_stats['positive_ratio'],
            'matrix_users': matrix_stats['num_users'],
            'matrix_products': matrix_stats['num_products'],
            'sentiment_auc': getattr(self.sentiment_analyzer, 'last_auc', 0.8),  # Default if not stored
            'total_recommendations': total_recommendations,
            'avg_predicted_rating': avg_predicted_rating
        }
    

    def run_complete_pipeline(self, sample_size=10000, num_users=3, recommendations_number=5, create_visualizations=True):
        """Run the complete pipeline with user customization"""
        print(" AMAZON SENTIMENT-AWARE RECOMMENDATION PIPELINE")
        print("=" * 60)
        print(f"Configuration:")
        print(f"   Dataset size: {sample_size:,} reviews")
        print(f"   Users for recommendations: {num_users}")
        print(f"   Recommendations per user: {recommendations_number}")
        print(f"   Create visualizations: {'Yes' if create_visualizations else 'No'}")
        print("=" * 60)
        
        try:
            self.setup_spark() \
                .load_and_explore_dataset(sample_size) \
                .perform_sentiment_analysis() \
                .create_rating_matrix() \
                .collaborative_filtering() \
                .show_recommendations(num_users, recommendations_number)
            
            print("\n Pipeline completed successfully!")
            
            # Show summary
            summary = self.get_pipeline_summary()
            if summary:
                print("\n Pipeline Summary:")
                print(f"    Dataset: {summary['total_reviews']:,} reviews, {summary['unique_users']:,} users, {summary['unique_products']:,} products")
                print(f"    Sentiment: {summary['positive_ratio']:.1%} positive reviews")
                print(f"    Matrix: {summary['matrix_users']:,} × {summary['matrix_products']:,}")
                print(f"    Recommendations: {summary['total_recommendations']} generated (avg rating: {summary['avg_predicted_rating']:.2f})")
            
            # Create visualizations
            if create_visualizations and summary:
                self.visualizer.generate_all_visualizations(self.df, self.recommendations_data, summary)
            
        except Exception as e:
            print(f"\n Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.spark_manager.stop_spark()
    
    def run_interactive_pipeline(self):
        """Run pipeline with interactive user input"""
        print("AMAZON SENTIMENT-AWARE RECOMMENDATION PIPELINE")
        print("=" * 70)
        
        # Get user input for dataset size
        while True:
            try:
                sample_size = input("Enter dataset size (1000-500000, default 5000): ").strip()
                if not sample_size:
                    sample_size = 5000
                else:
                    sample_size = int(sample_size)
                
                if 1000 <= sample_size <= 500000:
                    break
                else:
                    print(" Please enter a value between 1000 and 500000")
            except ValueError:
                print(" Please enter a valid number")
        
        # Get user input for number of users
        while True:
            try:
                num_users = input("Enter number of users for recommendations (1-10, default 3): ").strip()
                if not num_users:
                    num_users = 3
                else:
                    num_users = int(num_users)
                
                if 1 <= num_users <= 10:
                    break
                else:
                    print(" Please enter a value between 1 and 10")
            except ValueError:
                print(" Please enter a valid number")
        
        while True:
            rec_number = input("Enter number of recommendations per user (1-10, default 5): ").strip()
            if not rec_number:
                rec_number = 5
                break
            else:
                try:
                    rec_number = int(rec_number)
                    if 1 <= rec_number <= 10:
                        break
                    else:
                        print(" Please enter a value between 1 and 10")
                except ValueError:
                    print(" Please enter a valid number")

        # Ask about visualizations
        while True:
            viz_choice = input("Create visualizations? (y/n, default y): ").strip().lower()
            if not viz_choice or viz_choice in ['y', 'yes']:
                create_viz = True
                break
            elif viz_choice in ['n', 'no']:
                create_viz = False
                break
            else:
                print(" Please enter 'y' or 'n'")
        
        print(f"\n Running pipeline with:")
        print(f"   Dataset size: {sample_size:,}")
        print(f"   Users for recommendations: {num_users}")
        print(f"   Visualizations: {'Enabled' if create_viz else 'Disabled'}")
        
        input("\nPress Enter to start the pipeline...")
        
        # Run the pipeline
        self.run_complete_pipeline(sample_size, num_users, rec_number, create_viz)

def main():
    """Main entry point"""
    pipeline = PipelineOrchestrator()
    pipeline.run_complete_pipeline(sample_size=5000)

if __name__ == "__main__":
    main()
