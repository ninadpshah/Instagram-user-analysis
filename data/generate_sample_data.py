"""
Sample Data Generator for Social Media User Analysis
Run this script to generate sample data if you don't have the Kaggle dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_users: int = 5000) -> pd.DataFrame:
    """
    Generate realistic sample data matching the Kaggle dataset schema.

    Args:
        n_users: Number of users to generate

    Returns:
        DataFrame with sample social media user data
    """
    np.random.seed(42)

    # Define categories
    platforms = ['Instagram', 'Twitter', 'TikTok', 'YouTube', 'LinkedIn', 'Facebook']
    platform_weights = [0.30, 0.20, 0.20, 0.12, 0.10, 0.08]

    content_types = ['Image', 'Video', 'Text', 'Story', 'Reel', 'Live']
    content_weights = [0.25, 0.25, 0.15, 0.15, 0.12, 0.08]

    countries = ['United States', 'India', 'Brazil', 'Indonesia', 'United Kingdom',
                'Germany', 'France', 'Japan', 'Canada', 'Australia', 'Mexico',
                'South Korea', 'Spain', 'Italy', 'Netherlands']
    country_weights = [0.25, 0.18, 0.10, 0.08, 0.07, 0.05, 0.05, 0.04, 0.04, 0.04,
                      0.03, 0.03, 0.02, 0.01, 0.01]

    interests = ['Technology', 'Fashion', 'Travel', 'Food', 'Fitness', 'Music',
                'Gaming', 'Beauty', 'Sports', 'Photography', 'Art', 'Business',
                'Education', 'Comedy', 'Lifestyle']

    posting_frequencies = ['Multiple Daily', 'Daily', 'Weekly', 'Monthly']
    freq_weights = [0.15, 0.35, 0.35, 0.15]

    # Generate base data
    data = {
        'user_id': [f'USR_{i:06d}' for i in range(1, n_users + 1)],
        'username': [f'user_{np.random.randint(1000, 9999)}_{i}' for i in range(n_users)],
        'platform': np.random.choice(platforms, n_users, p=platform_weights),
        'age': np.random.normal(32, 12, n_users).clip(13, 75).astype(int),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_users, p=[0.48, 0.48, 0.04]),
        'country': np.random.choice(countries, n_users, p=country_weights),
        'content_type': np.random.choice(content_types, n_users, p=content_weights),
        'posting_frequency': np.random.choice(posting_frequencies, n_users, p=freq_weights),
        'peak_activity_hour': np.random.choice(range(24), n_users),
        'is_verified': np.random.choice([True, False], n_users, p=[0.08, 0.92]),
        'account_age_days': np.random.exponential(800, n_users).clip(30, 5000).astype(int),
    }

    df = pd.DataFrame(data)

    # Generate followers with realistic distribution (power law)
    base_followers = np.random.pareto(1.5, n_users) * 500
    df['followers'] = base_followers.clip(10, 50000000).astype(int)

    # Verified users have more followers
    df.loc[df['is_verified'], 'followers'] *= np.random.uniform(10, 100, df['is_verified'].sum())
    df['followers'] = df['followers'].astype(int)

    # Following count
    df['following'] = (df['followers'] * np.random.uniform(0.01, 2, n_users)).clip(0, 7500).astype(int)

    # Posts count based on account age and posting frequency
    freq_multiplier = df['posting_frequency'].map({
        'Multiple Daily': 3.0, 'Daily': 1.0, 'Weekly': 0.15, 'Monthly': 0.03
    })
    df['posts'] = (df['account_age_days'] * freq_multiplier * np.random.uniform(0.5, 1.5, n_users)).clip(1, 50000).astype(int)

    # Engagement metrics
    base_engagement = np.random.beta(2, 5, n_users) * 15
    follower_factor = np.log10(df['followers'] + 1) / 7
    df['avg_engagement_rate'] = (base_engagement * (1 - follower_factor * 0.5)).clip(0.1, 25).round(2)

    # Calculate likes, comments, shares
    avg_likes_per_post = df['followers'] * (df['avg_engagement_rate'] / 100) * np.random.uniform(0.5, 1.5, n_users)
    df['likes_received'] = (avg_likes_per_post * df['posts'] * 0.8).astype(int)
    df['comments_received'] = (df['likes_received'] * np.random.uniform(0.02, 0.15, n_users)).astype(int)
    df['shares_received'] = (df['likes_received'] * np.random.uniform(0.01, 0.08, n_users)).astype(int)

    # Total engagement
    df['total_engagement'] = df['likes_received'] + df['comments_received'] + df['shares_received']

    # Interests
    df['interests'] = ['|'.join(np.random.choice(interests, np.random.randint(1, 5), replace=False))
                      for _ in range(n_users)]
    df['primary_interest'] = [i.split('|')[0] for i in df['interests']]

    # Account creation date
    df['account_created'] = pd.to_datetime('2024-01-01') - pd.to_timedelta(df['account_age_days'], unit='D')

    # Last active
    df['last_active_days_ago'] = np.random.exponential(7, n_users).clip(0, 365).astype(int)
    df['last_active'] = pd.to_datetime('2024-01-01') - pd.to_timedelta(df['last_active_days_ago'], unit='D')

    return df


if __name__ == "__main__":
    print("Generating sample social media user data...")

    # Generate data
    df = generate_sample_data(5000)

    # Save to CSV
    output_path = Path(__file__).parent / 'social_media_users.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSample data generated successfully!")
    print(f"Output: {output_path}")
    print(f"Records: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
