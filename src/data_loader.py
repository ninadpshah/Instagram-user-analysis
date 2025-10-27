"""
Data Loading and Preprocessing Module for Social Media User Analysis
Author: Data Analytics Portfolio Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SocialMediaDataLoader:
    """
    A class to load and preprocess social media user data.
    Handles data validation, cleaning, and feature engineering.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the CSV data file. If None, uses sample data.
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Optional path override for the data file

        Returns:
            pandas DataFrame with loaded data
        """
        path = file_path or self.data_path
        if path is None:
            # Use sample data from the data directory
            path = Path(__file__).parent.parent / 'data' / 'sample_social_media_data.csv'

        self.raw_data = pd.read_csv(path)
        print(f"✓ Loaded {len(self.raw_data):,} records from {path}")
        return self.raw_data

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and validate the data.

        Args:
            df: DataFrame to clean. If None, uses loaded raw data.

        Returns:
            Cleaned DataFrame
        """
        data = df.copy() if df is not None else self.raw_data.copy()

        # Convert date columns
        date_columns = ['account_created', 'last_active']
        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')

        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

        # Clean string columns
        string_columns = data.select_dtypes(include=['object']).columns
        for col in string_columns:
            data[col] = data[col].fillna('Unknown').str.strip()

        print(f"✓ Data cleaned: {len(data):,} records")
        return data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for analysis.

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame with engineered features
        """
        data = df.copy()

        # Calculate account age in days
        if 'account_created' in data.columns:
            data['account_age_days'] = (pd.Timestamp.now() - data['account_created']).dt.days

        # Calculate engagement metrics
        if all(col in data.columns for col in ['likes_received', 'comments_received', 'shares_received', 'followers']):
            data['total_engagement'] = data['likes_received'] + data['comments_received'] + data['shares_received']
            data['engagement_per_follower'] = data['total_engagement'] / data['followers'].replace(0, 1)
            data['engagement_per_post'] = data['total_engagement'] / data['posts'].replace(0, 1)

        # Calculate follower to following ratio
        if all(col in data.columns for col in ['followers', 'following']):
            data['follower_ratio'] = data['followers'] / data['following'].replace(0, 1)

        # Categorize follower count
        if 'followers' in data.columns:
            data['follower_category'] = pd.cut(
                data['followers'],
                bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
                labels=['Nano (<1K)', 'Micro (1K-10K)', 'Mid-tier (10K-100K)',
                       'Macro (100K-1M)', 'Mega (>1M)']
            )

        # Categorize by age group
        if 'age' in data.columns:
            data['age_group'] = pd.cut(
                data['age'],
                bins=[0, 18, 25, 35, 45, 55, 100],
                labels=['<18', '18-24', '25-34', '35-44', '45-54', '55+']
            )

        # Extract primary interest
        if 'interests' in data.columns:
            data['primary_interest'] = data['interests'].str.split('|').str[0]

        # Posting activity level
        if 'posting_frequency' in data.columns:
            frequency_map = {
                'Multiple Daily': 'Very High',
                'Daily': 'High',
                'Weekly': 'Medium',
                'Monthly': 'Low'
            }
            data['activity_level'] = data['posting_frequency'].map(frequency_map).fillna('Medium')

        print(f"✓ Feature engineering complete: {len(data.columns)} features")
        self.processed_data = data
        return data

    def get_summary_stats(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Generate summary statistics for the dataset.

        Args:
            df: DataFrame to analyze. If None, uses processed data.

        Returns:
            Dictionary containing summary statistics
        """
        data = df if df is not None else self.processed_data

        summary = {
            'total_users': len(data),
            'total_followers': data['followers'].sum() if 'followers' in data.columns else 0,
            'avg_followers': data['followers'].mean() if 'followers' in data.columns else 0,
            'avg_engagement_rate': data['avg_engagement_rate'].mean() if 'avg_engagement_rate' in data.columns else 0,
            'platforms': data['platform'].nunique() if 'platform' in data.columns else 0,
            'countries': data['country'].nunique() if 'country' in data.columns else 0,
            'verified_users': data['is_verified'].sum() if 'is_verified' in data.columns else 0,
        }

        return summary

    def prepare_data(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Complete data preparation pipeline.

        Args:
            file_path: Optional path to data file

        Returns:
            Tuple of (processed DataFrame, summary statistics)
        """
        self.load_data(file_path)
        cleaned = self.clean_data()
        processed = self.engineer_features(cleaned)
        summary = self.get_summary_stats()

        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value:,}")
        print("="*50)

        return processed, summary


if __name__ == "__main__":
    # Test the data loader
    loader = SocialMediaDataLoader()
    df, stats = loader.prepare_data()
    print("\nSample data:")
    print(df.head())
