import sys
import argparse
from pipeline.main_pipeline import PipelineOrchestrator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Amazon Sentiment-Aware Recommendation Pipeline')
    
    parser.add_argument('--size', type=int, default=None,
                       help='Dataset size (1000-1000000, default: interactive mode)')
    parser.add_argument('--users', type=int, default=3,
                       help='Number of users for recommendations (1-10, default: 3)')
    parser.add_argument('--recommendations', type=int, default=5,
                       help='Number of recommendations per user (default: 5)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualizations')
    parser.add_argument('--interactive', action='store_true',
                       help='Force interactive mode')
    
    return parser.parse_args()

def validate_inputs(sample_size, num_users):
    """Validate input parameters"""
    if not (1000 <= sample_size <= 1000000):
        print(f"Dataset size must be between 1000 and 1000000 (got {sample_size})")
        return False
    
    if not (1 <= num_users <= 10):
        print(f"Number of users must be between 1 and 10 (got {num_users})")
        return False
    
    return True

def run_direct_mode(sample_size, num_users, create_viz):
    """Run pipeline in direct mode with specified parameters"""
    print("AMAZON SENTIMENT-AWARE RECOMMENDATION PIPELINE")
    print("=" * 65)
    print("This pipeline includes:")
    print("1. Spark Context setup")
    print("2. Dataset loading and exploration")
    print("3. MLlib sentiment analysis")
    print("4. Rating matrix creation from sentiment")
    print("5. Collaborative filtering recommendations")
    if create_viz:
        print("6. Visualization generation")
    print("=" * 65)
    
    if not validate_inputs(sample_size, num_users):
        return
    
    # Create and run pipeline
    pipeline = PipelineOrchestrator()
    pipeline.run_complete_pipeline(sample_size, num_users, create_viz)

def run_interactive_mode():
    """Run pipeline in interactive mode"""
    pipeline = PipelineOrchestrator()
    pipeline.run_interactive_pipeline()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Interactive mode
    if args.interactive or args.size is None:
        run_interactive_mode()
    else:
        # Direct mode
        sample_size = args.size
        num_users = args.users
        create_viz = not args.no_viz
        
        run_direct_mode(sample_size, num_users, create_viz)

if __name__ == "__main__":
    main()
