"""
Dataset Generator for URL Credibility Neural Network
Generates labeled training data using Claude and existing heuristics
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deliverable1 as d1
from nn.feature_extractor import URLFeatureExtractor
from nn.web_search import WebSearchAnalyzer
import numpy as np
from datetime import datetime


class DatasetGenerator:
    """Generate training datasets for URL credibility prediction"""

    def __init__(self, api_key: Optional[str] = None):
        self.feature_extractor = URLFeatureExtractor()
        self.web_analyzer = WebSearchAnalyzer(api_key)
        self.dataset = []

    def generate_from_search_queries(self, queries: List[str], urls_per_query: int = 10) -> pd.DataFrame:
        """
        Generate dataset by searching for topics and analyzing results

        Args:
            queries: List of search queries to find diverse URLs
            urls_per_query: Number of URLs to analyze per query
        """
        print(f"Generating dataset from {len(queries)} search queries...")

        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing query: '{query}'")

            # Search and get URLs
            search_results = self.web_analyzer.search_and_analyze(query, urls_per_query)

            for result in search_results:
                url = result['url']
                print(f"  Processing: {url}")

                # Extract features
                heuristic_scores = self._get_heuristic_scores(url)
                features = self.feature_extractor.extract_all_features(url, heuristic_scores)

                # Add Claude scores as features
                features['claude_authority'] = result['claude_scores']['authority']
                features['claude_quality'] = result['claude_scores']['quality']
                features['claude_evidence'] = result['claude_scores']['evidence']
                features['claude_objectivity'] = result['claude_scores']['objectivity']
                features['claude_recency'] = result['claude_scores']['recency']
                features['claude_overall'] = result['credibility_score']

                # Add labels
                features['credibility_score'] = result['credibility_score']
                features['credibility_class'] = result['credibility_class']
                features['credibility_label'] = result['credibility_label']
                features['url'] = url
                features['query'] = query

                self.dataset.append(features)

        df = pd.DataFrame(self.dataset)
        print(f"\n✅ Generated dataset with {len(df)} samples")
        return df

    def generate_from_url_list(self, urls: List[str]) -> pd.DataFrame:
        """
        Generate dataset from a list of URLs

        Args:
            urls: List of URLs to analyze and label
        """
        print(f"Generating dataset from {len(urls)} URLs...")

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing: {url}")

            try:
                # Get heuristic scores
                heuristic_scores = self._get_heuristic_scores(url)

                # Extract features
                features = self.feature_extractor.extract_all_features(url, heuristic_scores)

                # Get Claude labeling
                page_data = self.web_analyzer.fetch_page_content(url)
                if page_data['success']:
                    label_data = self.web_analyzer.create_training_label(
                        url,
                        f"Title: {page_data['title']}\n\n{page_data['content']}"
                    )

                    # Add Claude scores and labels
                    features['claude_authority'] = label_data['claude_scores']['authority']
                    features['claude_quality'] = label_data['claude_scores']['quality']
                    features['claude_evidence'] = label_data['claude_scores']['evidence']
                    features['claude_objectivity'] = label_data['claude_scores']['objectivity']
                    features['claude_recency'] = label_data['claude_scores']['recency']
                    features['claude_overall'] = label_data['credibility_score']

                    features['credibility_score'] = label_data['credibility_score']
                    features['credibility_class'] = label_data['credibility_class']
                    features['credibility_label'] = label_data['credibility_label']
                    features['url'] = url

                    self.dataset.append(features)
                else:
                    print(f"  ⚠️  Failed to fetch content: {page_data.get('error', 'Unknown')}")

            except Exception as e:
                print(f"  ❌ Error processing {url}: {str(e)}")

        df = pd.DataFrame(self.dataset)
        print(f"\n✅ Generated dataset with {len(df)} samples")
        return df

    def _get_heuristic_scores(self, url: str) -> Dict:
        """Get heuristic scores from deliverable1"""
        try:
            pagerank_scores = d1.compute_pagerank([url])
            scores = d1.score_url_with_content(url, pagerank_scores)
            return scores
        except Exception as e:
            print(f"  ⚠️  Heuristic scoring failed: {str(e)}")
            return {
                'url_score': 0.5,
                'text_score': 0.5,
                'popularity_score': 0.5,
                'pagerank_score': 0.5,
                'combined_score': 0.5
            }

    def save_dataset(self, df: pd.DataFrame, filename: str = None):
        """Save dataset to CSV and JSON"""
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/credibility_dataset_{timestamp}"
        elif not filename.startswith("data/"):
            filename = f"data/{filename}"

        # Save as CSV
        csv_path = f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved dataset to {csv_path}")

        # Save as JSON for backup
        json_path = f"{filename}.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"💾 Saved dataset to {json_path}")

        return csv_path, json_path

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load existing dataset"""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError("File must be CSV or JSON")

    def augment_with_balanced_samples(self, df: pd.DataFrame, target_per_class: int = 100) -> pd.DataFrame:
        """
        Augment dataset to balance classes by searching for more examples
        """
        print("\n📊 Checking class distribution...")

        # Count samples per class
        class_counts = df['credibility_class'].value_counts().to_dict()
        print(f"Current distribution: {class_counts}")

        # Queries for each credibility level
        queries_by_class = {
            2: [  # High credibility
                "scientific research papers",
                "government statistics",
                "university research findings",
                "peer reviewed journals",
                "national library resources"
            ],
            1: [  # Medium credibility
                "blog posts about technology",
                "personal finance advice",
                "how to guides",
                "product reviews",
                "news commentary"
            ],
            0: [  # Low credibility
                "conspiracy theories",
                "unverified claims",
                "clickbait articles",
                "sensational headlines",
                "sponsored content"
            ]
        }

        # Generate additional samples for underrepresented classes
        new_samples = []
        for cls, count in class_counts.items():
            needed = target_per_class - count
            if needed > 0:
                print(f"\n🔍 Need {needed} more samples for class {cls}")
                queries = queries_by_class.get(cls, [])
                per_query = max(1, needed // len(queries))

                for query in queries[:min(len(queries), needed)]:
                    results = self.web_analyzer.search_and_analyze(query, per_query)

                    for result in results:
                        if result['credibility_class'] == cls:
                            # Process this URL
                            url = result['url']
                            heuristic_scores = self._get_heuristic_scores(url)
                            features = self.feature_extractor.extract_all_features(url, heuristic_scores)

                            features.update({
                                'claude_authority': result['claude_scores']['authority'],
                                'claude_quality': result['claude_scores']['quality'],
                                'claude_evidence': result['claude_scores']['evidence'],
                                'claude_objectivity': result['claude_scores']['objectivity'],
                                'claude_recency': result['claude_scores']['recency'],
                                'claude_overall': result['credibility_score'],
                                'credibility_score': result['credibility_score'],
                                'credibility_class': result['credibility_class'],
                                'credibility_label': result['credibility_label'],
                                'url': url,
                                'query': query
                            })

                            new_samples.append(features)

        if new_samples:
            augmented_df = pd.concat([df, pd.DataFrame(new_samples)], ignore_index=True)
            print(f"\n✅ Augmented dataset from {len(df)} to {len(augmented_df)} samples")
            return augmented_df
        else:
            print("\n✅ Dataset already balanced")
            return df


def create_sample_dataset():
    """Create a sample dataset with diverse URLs"""

    # Sample diverse search queries to get varied URLs
    search_queries = [
        # High credibility sources
        "latest scientific research findings",
        "government health guidelines",
        "university academic publications",

        # Medium credibility sources
        "technology news updates",
        "expert blog posts",

        # Mixed/lower credibility
        "controversial health claims",
        "product marketing pages",
    ]

    generator = DatasetGenerator()

    # Generate from searches
    df = generator.generate_from_search_queries(search_queries, urls_per_query=5)

    # Save dataset
    generator.save_dataset(df, "credibility_training_data")

    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['credibility_label'].value_counts())
    print(f"\nCredibility score distribution:")
    print(df['credibility_score'].describe())

    return df


if __name__ == "__main__":
    # Create sample dataset
    print("🚀 Creating sample training dataset...\n")
    df = create_sample_dataset()
    print(f"\n✅ Dataset created successfully with {len(df)} samples!")
