"""
Trend Analysis Module for Social Media User Analysis
Identifies patterns, trends, and insights from user data
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TrendAnalyzer:
    """
    A class for analyzing trends and patterns in social media user data.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the trend analyzer.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.insights = []

    def analyze_engagement_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze engagement patterns and trends.
        """
        trends = {}

        # Engagement by platform
        platform_engagement = df.groupby('platform').agg({
            'avg_engagement_rate': ['mean', 'median', 'std'],
            'followers': 'mean',
            'user_id': 'count'
        }).round(2)

        trends['platform_engagement'] = platform_engagement

        # Engagement by content type
        content_engagement = df.groupby('content_type').agg({
            'avg_engagement_rate': ['mean', 'median'],
            'likes_received': 'mean',
            'comments_received': 'mean'
        }).round(2)

        trends['content_engagement'] = content_engagement

        # Engagement by follower category
        if 'follower_category' in df.columns:
            follower_engagement = df.groupby('follower_category')['avg_engagement_rate'].agg(['mean', 'median', 'count'])
            trends['follower_engagement'] = follower_engagement

            # Key insight: engagement tends to decrease with more followers
            correlation = df['followers'].corr(df['avg_engagement_rate'])
            if correlation < -0.1:
                self.insights.append(f"Engagement Paradox: Users with fewer followers have higher engagement rates (correlation: {correlation:.2f})")

        return trends

    def analyze_platform_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze platform-specific trends and characteristics.
        """
        trends = {}

        # Platform demographics
        platform_demo = df.groupby('platform').agg({
            'age': 'mean',
            'is_verified': 'mean',
            'followers': ['mean', 'median'],
            'posts': 'mean'
        }).round(2)

        trends['platform_demographics'] = platform_demo

        # Platform content preferences
        platform_content = pd.crosstab(df['platform'], df['content_type'], normalize='index') * 100
        trends['platform_content_preferences'] = platform_content.round(2)

        # Find dominant content type per platform
        dominant_content = platform_content.idxmax(axis=1)
        self.insights.append(f"Content Preferences: " + ", ".join([f"{p} favors {c}" for p, c in dominant_content.items()]))

        return trends

    def analyze_demographic_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze demographic patterns in the data.
        """
        trends = {}

        # Age analysis
        if 'age_group' in df.columns:
            age_analysis = df.groupby('age_group').agg({
                'followers': 'mean',
                'avg_engagement_rate': 'mean',
                'posts': 'mean',
                'user_id': 'count'
            }).round(2)
            trends['age_analysis'] = age_analysis

            # Find most engaged age group
            most_engaged_age = df.groupby('age_group')['avg_engagement_rate'].mean().idxmax()
            self.insights.append(f"Most Engaged Demographics: {most_engaged_age} age group shows highest engagement")

        # Gender analysis
        gender_analysis = df.groupby('gender').agg({
            'followers': ['mean', 'median'],
            'avg_engagement_rate': 'mean',
            'user_id': 'count'
        }).round(2)
        trends['gender_analysis'] = gender_analysis

        # Country analysis
        country_analysis = df.groupby('country').agg({
            'followers': 'mean',
            'avg_engagement_rate': 'mean',
            'user_id': 'count'
        }).sort_values(('user_id', ), ascending=False).head(10).round(2)
        trends['country_analysis'] = country_analysis

        return trends
