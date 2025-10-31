"""
Training Script for URL Credibility Neural Network
Generates dataset and trains the model
"""

import argparse
import os
import sys

# Add parent directory to path to import nn package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.dataset_generator import DatasetGenerator
from nn.credibility_nn import CredibilityPredictor
import pandas as pd

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def train_from_search_queries(queries, urls_per_query=10, epochs=100):
    """
    Train model by searching for topics and analyzing results

    Args:
        queries: List of search queries
        urls_per_query: Number of URLs to analyze per query
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("URL Credibility Neural Network - Training Pipeline")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n📊 STEP 1: Generating Training Dataset")
    print("-" * 60)

    generator = DatasetGenerator()
    df = generator.generate_from_search_queries(queries, urls_per_query)

    # Save dataset
    dataset_path, _ = generator.save_dataset(df, "credibility_training_data")

    # Step 2: Train model
    print("\n🧠 STEP 2: Training Neural Network")
    print("-" * 60)

    predictor = CredibilityPredictor()
    results = predictor.train(df, epochs=epochs)

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Dataset: {dataset_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution:")
    print(df['credibility_label'].value_counts())

    print(f"\n🧠 Model Performance:")
    print(f"   Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"   Test accuracy: {results['test_acc']:.4f}")

    print(f"\n💾 Model saved to: credibility_model.pth")

    return predictor, df


def train_from_url_list(url_file, epochs=100):
    """
    Train model from a file containing URLs

    Args:
        url_file: Path to file with URLs (one per line)
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("URL Credibility Neural Network - Training from URL List")
    print("=" * 60)

    # Load URLs (skip comments and empty lines)
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')]

    print(f"\n📋 Loaded {len(urls)} URLs from {url_file}")

    # Generate dataset
    print("\n📊 STEP 1: Generating Training Dataset")
    print("-" * 60)

    generator = DatasetGenerator()
    df = generator.generate_from_url_list(urls)

    # Save dataset
    dataset_path, _ = generator.save_dataset(df, "credibility_training_data")

    # Train model
    print("\n🧠 STEP 2: Training Neural Network")
    print("-" * 60)

    predictor = CredibilityPredictor()
    results = predictor.train(df, epochs=epochs)

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Dataset: {dataset_path}")
    print(f"   Total samples: {len(df)}")

    print(f"\n🧠 Model Performance:")
    print(f"   Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"   Test accuracy: {results['test_acc']:.4f}")

    print(f"\n💾 Model saved to: credibility_model.pth")

    return predictor, df


def train_from_existing_dataset(dataset_file, epochs=100):
    """
    Train model from existing dataset CSV/JSON

    Args:
        dataset_file: Path to dataset file
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("URL Credibility Neural Network - Training from Dataset")
    print("=" * 60)

    # Load dataset
    print(f"\n📊 Loading dataset from {dataset_file}")
    generator = DatasetGenerator()
    df = generator.load_dataset(dataset_file)

    print(f"   Loaded {len(df)} samples")
    print(f"   Class distribution:")
    print(df['credibility_label'].value_counts())

    # Train model
    print("\n🧠 Training Neural Network")
    print("-" * 60)

    predictor = CredibilityPredictor()
    results = predictor.train(df, epochs=epochs)

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    print(f"\n🧠 Model Performance:")
    print(f"   Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"   Test accuracy: {results['test_acc']:.4f}")

    print(f"\n💾 Model saved to: credibility_model.pth")

    return predictor, df


def quick_demo():
    """Quick demo with sample dataset"""
    print("=" * 60)
    print("URL Credibility Neural Network - Quick Demo")
    print("=" * 60)

    # Use predefined diverse queries
    queries = [
        # High credibility
        "scientific research climate change",
        "government health guidelines COVID",
        "university research artificial intelligence",

        # Medium credibility
        "tech news latest developments",
        "expert opinions machine learning",

        # Lower credibility (for balanced dataset)
        "health myths debunked",
        "controversial claims social media",
    ]

    return train_from_search_queries(queries, urls_per_query=20, epochs=50)  # Increased to 5 URLs per query


def main():
    parser = argparse.ArgumentParser(description='Train URL Credibility Neural Network')

    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'search', 'urls', 'dataset'],
                       help='Training mode: demo, search, urls, or dataset')

    parser.add_argument('--queries', type=str, nargs='+',
                       help='Search queries for search mode')

    parser.add_argument('--urls-file', type=str,
                       help='File containing URLs (one per line) for urls mode')

    parser.add_argument('--dataset', type=str,
                       help='Path to existing dataset CSV/JSON for dataset mode')

    parser.add_argument('--urls-per-query', type=int, default=10,
                       help='Number of URLs to analyze per query (default: 10)')

    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')

    args = parser.parse_args()

    if args.mode == 'demo':
        print("🚀 Running quick demo...\n")
        predictor, df = quick_demo()

    elif args.mode == 'search':
        if not args.queries:
            print("❌ Error: --queries required for search mode")
            return
        predictor, df = train_from_search_queries(
            args.queries, args.urls_per_query, args.epochs
        )

    elif args.mode == 'urls':
        if not args.urls_file:
            print("❌ Error: --urls-file required for urls mode")
            return
        predictor, df = train_from_url_list(args.urls_file, args.epochs)

    elif args.mode == 'dataset':
        if not args.dataset:
            print("❌ Error: --dataset required for dataset mode")
            return
        predictor, df = train_from_existing_dataset(args.dataset, args.epochs)

    print("\n" + "=" * 60)
    print("🎉 All done! You can now use the trained model in your app.")
    print("=" * 60)


if __name__ == "__main__":
    # If no arguments provided, run demo
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick demo...\n")
        quick_demo()
    else:
        main()
