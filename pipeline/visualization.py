#!/usr/bin/env python3
"""
Visualization Module

Provides simple visualizations for sentiment analysis and collaborative filtering results.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for compatibility
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import col
import os

class PipelineVisualizer:
    """Handles visualization of pipeline results"""

    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')

    def __init__(self, output_dir=f"output/pipeline_results-{timestamp}"):
        self.output_dir = output_dir
        plt.style.use('default')
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def visualize_dataset_overview(self, df):
        """Create dataset overview visualizations"""
        print("   Creating dataset overview plots...")
        self.ensure_output_dir()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Rating Distribution
        rating_data = df.groupBy('rating').count().orderBy('rating').collect()
        ratings = [row.rating for row in rating_data]
        counts = [row['count'] for row in rating_data]  # Use dictionary access instead of attribute access
        
        ax1.bar(ratings, counts, color='skyblue', alpha=0.7)
        ax1.set_title('Rating Distribution')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Number of Reviews')
        ax1.grid(True, alpha=0.3)
        
        # 2. Text Length Distribution
        text_lengths = df.select('text').rdd.map(lambda row: len(row.text) if row.text else 0).collect()
        ax2.hist(text_lengths, bins=30, color='lightgreen', alpha=0.7)
        ax2.set_title('Review Text Length Distribution')
        ax2.set_xlabel('Text Length (characters)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top Users by Review Count
        user_counts = df.groupBy('user_id').count().orderBy(col('count').desc()).limit(10).collect()
        user_labels = [f"User {i+1}" for i in range(len(user_counts))]
        user_review_counts = [row['count'] for row in user_counts]  # Use dictionary access
        
        ax3.barh(user_labels, user_review_counts, color='orange', alpha=0.7)
        ax3.set_title('Top 10 Users by Review Count')
        ax3.set_xlabel('Number of Reviews')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top Products by Review Count
        product_counts = df.groupBy('parent_asin').count().orderBy(col('count').desc()).limit(10).collect()
        product_labels = [f"Product {i+1}" for i in range(len(product_counts))]
        product_review_counts = [row['count'] for row in product_counts]  # Use dictionary access
        
        ax4.barh(product_labels, product_review_counts, color='coral', alpha=0.7)
        ax4.set_title('Top 10 Products by Review Count')
        ax4.set_xlabel('Number of Reviews')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dataset_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Dataset overview saved to {self.output_dir}/dataset_overview.png")
    
    def visualize_sentiment_analysis(self, df):
        """Create sentiment analysis visualizations"""
        print("   Creating sentiment analysis plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Label Distribution
        sentiment_data = df.groupBy('sentiment_label').count().collect()
        labels = ['Negative' if row.sentiment_label == 0.0 else 'Positive' for row in sentiment_data]
        counts = [row['count'] for row in sentiment_data]  # Use dictionary access
        colors = ['red' if label == 'Negative' else 'green' for label in labels]
        
        ax1.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Sentiment Distribution')
        
        # 2. Sentiment Score Distribution
        sentiment_scores = df.select('sentiment_score').rdd.map(lambda row: row.sentiment_score).collect()
        ax2.hist(sentiment_scores, bins=30, color='purple', alpha=0.7)
        ax2.set_title('Sentiment Score Distribution')
        ax2.set_xlabel('Sentiment Score (0=Negative, 1=Positive)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rating vs Sentiment Score
        rating_sentiment = df.select('rating', 'sentiment_score').collect()
        ratings = [row.rating for row in rating_sentiment]
        scores = [row.sentiment_score for row in rating_sentiment]
        
        ax3.scatter(ratings, scores, alpha=0.5, color='blue')
        ax3.set_title('Original Rating vs Sentiment Score')
        ax3.set_xlabel('Original Rating')
        ax3.set_ylabel('Sentiment Score')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(ratings, scores, 1)
        p = np.poly1d(z)
        ax3.plot(ratings, p(ratings), "r--", alpha=0.8)
        
        # 4. Enhanced vs Original Rating
        enhanced_ratings = df.select('rating', 'enhanced_rating').collect()
        original = [row.rating for row in enhanced_ratings]
        enhanced = [row.enhanced_rating for row in enhanced_ratings]
        
        ax4.scatter(original, enhanced, alpha=0.5, color='green')
        ax4.plot([1, 5], [1, 5], 'r--', alpha=0.8, label='No change line')
        ax4.set_title('Original vs Enhanced Rating')
        ax4.set_xlabel('Original Rating')
        ax4.set_ylabel('Enhanced Rating')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sentiment_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Sentiment analysis plots saved to {self.output_dir}/sentiment_analysis.png")
    
    def visualize_collaborative_filtering(self, recommendations_data):
        """Create collaborative filtering visualizations"""
        print("   Creating collaborative filtering plots...")
        
        if not recommendations_data:
            print("     No recommendations data to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Collaborative Filtering Results', fontsize=16, fontweight='bold')
        
        # Extract data from recommendations
        all_ratings = []
        user_rec_counts = {}
        
        for user_id, recs in recommendations_data.items():
            user_rec_counts[user_id] = len(recs)
            for rec in recs:
                all_ratings.append(rec['final_rating'])
        
        # 1. Predicted Rating Distribution
        ax1.hist(all_ratings, bins=20, color='gold', alpha=0.7)
        ax1.set_title('Predicted Rating Distribution')
        ax1.set_xlabel('Predicted Rating')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Recommendations per User
        users = list(user_rec_counts.keys())
        rec_counts = list(user_rec_counts.values())
        user_labels = [f"User {i+1}" for i in range(len(users))]
        
        ax2.bar(user_labels, rec_counts, color='lightblue', alpha=0.7)
        ax2.set_title('Number of Recommendations per User')
        ax2.set_xlabel('Users')
        ax2.set_ylabel('Number of Recommendations')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rating Quality Distribution (bins)
        rating_bins = ['1.0-2.0', '2.0-3.0', '3.0-4.0', '4.0-5.0']
        bin_counts = [0, 0, 0, 0]
        
        for rating in all_ratings:
            if rating < 2.0:
                bin_counts[0] += 1
            elif rating < 3.0:
                bin_counts[1] += 1
            elif rating < 4.0:
                bin_counts[2] += 1
            else:
                bin_counts[3] += 1
        
        colors = ['red', 'orange', 'yellow', 'green']
        ax3.bar(rating_bins, bin_counts, color=colors, alpha=0.7)
        ax3.set_title('Recommendation Quality Distribution')
        ax3.set_xlabel('Predicted Rating Range')
        ax3.set_ylabel('Number of Recommendations')
        ax3.grid(True, alpha=0.3)
        
        # 4. Average Rating per User
        avg_ratings = []
        for user_id, recs in recommendations_data.items():
            if recs:
                avg_rating = sum(rec['final_rating'] for rec in recs) / len(recs)
                avg_ratings.append(avg_rating)
        
        ax4.bar(user_labels[:len(avg_ratings)], avg_ratings, color='mediumseagreen', alpha=0.7)
        ax4.set_title('Average Predicted Rating per User')
        ax4.set_xlabel('Users')
        ax4.set_ylabel('Average Predicted Rating')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/collaborative_filtering.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Collaborative filtering plots saved to {self.output_dir}/collaborative_filtering.png")
    
    def create_pipeline_summary(self, stats):
        """Create a summary visualization of the entire pipeline"""
        print("   Creating pipeline summary...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pipeline Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Dataset Statistics
        dataset_labels = ['Total Reviews', 'Unique Users', 'Unique Products']
        dataset_values = [
            stats.get('total_reviews', 0),
            stats.get('unique_users', 0), 
            stats.get('unique_products', 0)
        ]
        
        ax1.bar(dataset_labels, dataset_values, color=['skyblue', 'lightgreen', 'coral'], alpha=0.7)
        ax1.set_title('Dataset Statistics')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(dataset_values):
            ax1.text(i, v + max(dataset_values) * 0.01, str(v), ha='center', va='bottom')
        
        # 2. Sentiment Analysis Performance
        sentiment_labels = ['Positive', 'Negative']
        sentiment_values = [
            stats.get('positive_reviews', 0),
            stats.get('negative_reviews', 0)
        ]
        
        ax2.pie(sentiment_values, labels=sentiment_labels, colors=['green', 'red'], 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Sentiment Analysis Results')
        
        # 3. Matrix Statistics
        matrix_labels = ['Users', 'Products', 'Sparsity %']
        matrix_values = [
            stats.get('matrix_users', 0),
            stats.get('matrix_products', 0),
            stats.get('matrix_sparsity', 0) * 100  # Convert to percentage
        ]
        
        ax3.bar(matrix_labels, matrix_values, color=['orange', 'purple', 'brown'], alpha=0.7)
        ax3.set_title('Rating Matrix Statistics')
        ax3.set_ylabel('Count / Percentage')
        
        # 4. Model Performance
        performance_labels = ['AUC Score', 'Recommendations\nGenerated', 'Avg Rating']
        performance_values = [
            stats.get('sentiment_auc', 0) * 100,  # Convert to percentage
            stats.get('total_recommendations', 0),
            stats.get('avg_predicted_rating', 0)
        ]
        
        bars = ax4.bar(performance_labels, performance_values, 
                      color=['gold', 'lightblue', 'lightcoral'], alpha=0.7)
        ax4.set_title('Model Performance')
        ax4.set_ylabel('Score / Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(performance_values) * 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pipeline_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Pipeline summary saved to {self.output_dir}/pipeline_summary.png")

    def create_recommedations_csv(self, recommendations_data):
        """Create CSV file for recommendations"""
        print("   Creating recommendations CSV...")
        
        import pandas as pd
        
        if not recommendations_data:
            print("   No recommendations data to save")
            return
        
        recs_list = []
        for user_id, recs in recommendations_data.items():
            for rec in recs:
                recs_list.append({
                    'user_id': user_id,
                    'product_id': rec['product_id'],
                    'product_title': rec.get('product_title', 'Unknown Product'),
                    'predicted_rating': rec['final_rating'],
                    'sentiment_score': rec['sentiment_score'],
                })
        
        df_recs = pd.DataFrame(recs_list)
        df_recs.to_csv(f"{self.output_dir}/recommendations.csv", index=False)
        
        print(f"    Recommendations saved to {self.output_dir}/recommendations.csv")
    
    def generate_all_visualizations(self, df, recommendations_data, stats):
        """Generate all visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        try:
            self.visualize_dataset_overview(df)
            self.visualize_sentiment_analysis(df)
            self.visualize_collaborative_filtering(recommendations_data)
            self.create_pipeline_summary(stats)
            self.create_recommedations_csv(recommendations_data)
            
            print(f"\n All visualizations created")
            print(f" Results saved in: {os.path.abspath(self.output_dir)}/")
            print("   Files created:")
            print("   - dataset_overview.png")
            print("   - sentiment_analysis.png") 
            print("   - collaborative_filtering.png")
            print("   - pipeline_summary.png")
            print("   - recommendations.csv")
            
        except Exception as e:
            print(f" Visualization error: {e}")
            import traceback
            traceback.print_exc()
