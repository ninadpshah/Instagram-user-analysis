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

    def analyze_activity_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze user activity patterns.
        """
        trends = {}

        # Peak hours analysis
        hour_analysis = df.groupby('peak_activity_hour').agg({
            'avg_engagement_rate': 'mean',
            'user_id': 'count'
        }).round(2)
        trends['hour_analysis'] = hour_analysis

        # Find best posting times
        best_hours = hour_analysis['avg_engagement_rate'].nlargest(3).index.tolist()
        self.insights.append(f"Optimal Posting Times: {', '.join([f'{h}:00' for h in best_hours])} show highest engagement")

        # Posting frequency analysis
        freq_analysis = df.groupby('posting_frequency').agg({
            'avg_engagement_rate': 'mean',
            'followers': 'mean',
            'user_id': 'count'
        }).round(2)
        trends['frequency_analysis'] = freq_analysis

        # Activity level analysis
        if 'activity_level' in df.columns:
            activity_analysis = df.groupby('activity_level').agg({
                'avg_engagement_rate': 'mean',
                'followers': 'mean'
            }).round(2)
            trends['activity_analysis'] = activity_analysis

        return trends

    def analyze_verified_vs_unverified(self, df: pd.DataFrame) -> Dict:
        """
        Compare verified and unverified users.
        """
        comparison = df.groupby('is_verified').agg({
            'followers': ['mean', 'median', 'sum'],
            'avg_engagement_rate': ['mean', 'median'],
            'posts': 'mean',
            'user_id': 'count'
        }).round(2)

        # Statistical test
        verified = df[df['is_verified']]['avg_engagement_rate']
        unverified = df[~df['is_verified']]['avg_engagement_rate']

        if len(verified) > 0 and len(unverified) > 0:
            t_stat, p_value = stats.ttest_ind(verified, unverified)
            if p_value < 0.05:
                direction = "higher" if verified.mean() > unverified.mean() else "lower"
                self.insights.append(f"Verification Impact: Verified users have statistically {direction} engagement (p={p_value:.4f})")

        return {'comparison': comparison}

    def identify_top_performers(self, df: pd.DataFrame, n: int = 10) -> Dict:
        """
        Identify top performing users across different metrics.
        """
        top_performers = {}

        # By followers
        top_performers['by_followers'] = df.nlargest(n, 'followers')[
            ['username', 'platform', 'followers', 'avg_engagement_rate', 'is_verified']
        ]

        # By engagement rate (minimum followers threshold)
        qualified = df[df['followers'] >= 1000]
        top_performers['by_engagement'] = qualified.nlargest(n, 'avg_engagement_rate')[
            ['username', 'platform', 'followers', 'avg_engagement_rate', 'is_verified']
        ]

        # By total engagement
        top_performers['by_total_engagement'] = df.nlargest(n, 'total_engagement')[
            ['username', 'platform', 'followers', 'total_engagement', 'avg_engagement_rate']
        ]

        return top_performers

    def plot_trend_dashboard(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create comprehensive trend analysis dashboard.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Engagement by Platform', 'Engagement by Content Type', 'Engagement by Age Group',
                'Peak Activity Hours', 'Verified vs Non-Verified', 'Posting Frequency Impact',
                'Follower Category Engagement', 'Geographic Distribution', 'Activity Level Analysis'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'pie'}]
            ]
        )

        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']

        # 1. Engagement by Platform
        platform_eng = df.groupby('platform')['avg_engagement_rate'].mean().sort_values(ascending=True)
        fig.add_trace(go.Bar(
            y=platform_eng.index, x=platform_eng.values,
            orientation='h', marker_color='#667eea'
        ), row=1, col=1)

        # 2. Engagement by Content Type
        content_eng = df.groupby('content_type')['avg_engagement_rate'].mean().sort_values(ascending=True)
        fig.add_trace(go.Bar(
            y=content_eng.index, x=content_eng.values,
            orientation='h', marker_color='#764ba2'
        ), row=1, col=2)

        # 3. Engagement by Age Group
        if 'age_group' in df.columns:
            age_eng = df.groupby('age_group')['avg_engagement_rate'].mean()
            fig.add_trace(go.Bar(
                x=age_eng.index.astype(str), y=age_eng.values,
                marker_color='#f093fb'
            ), row=1, col=3)

        # 4. Peak Activity Hours
        hour_data = df.groupby('peak_activity_hour').agg({
            'avg_engagement_rate': 'mean',
            'user_id': 'count'
        }).reset_index()
        fig.add_trace(go.Scatter(
            x=hour_data['peak_activity_hour'],
            y=hour_data['avg_engagement_rate'],
            mode='lines+markers',
            line=dict(color='#f5576c', width=3),
            marker=dict(size=8)
        ), row=2, col=1)

        # 5. Verified vs Non-Verified
        verified_data = df.groupby('is_verified')['avg_engagement_rate'].mean()
        verified_data.index = verified_data.index.map({True: 'Verified', False: 'Not Verified'})
        fig.add_trace(go.Bar(
            x=verified_data.index, y=verified_data.values,
            marker_color=['#667eea', '#f5576c']
        ), row=2, col=2)

        # 6. Posting Frequency Impact
        freq_eng = df.groupby('posting_frequency')['avg_engagement_rate'].mean()
        fig.add_trace(go.Bar(
            x=freq_eng.index, y=freq_eng.values,
            marker_color='#4facfe'
        ), row=2, col=3)

        # 7. Follower Category Engagement
        if 'follower_category' in df.columns:
            cat_eng = df.groupby('follower_category')['avg_engagement_rate'].mean()
            fig.add_trace(go.Bar(
                x=cat_eng.index.astype(str), y=cat_eng.values,
                marker_color='#00f2fe'
            ), row=3, col=1)

        # 8. Geographic Distribution (Top 10)
        country_users = df['country'].value_counts().head(10)
        fig.add_trace(go.Bar(
            y=country_users.index, x=country_users.values,
            orientation='h', marker_color='#43e97b'
        ), row=3, col=2)

        # 9. Activity Level Distribution
        if 'activity_level' in df.columns:
            activity_dist = df['activity_level'].value_counts()
            fig.add_trace(go.Pie(
                labels=activity_dist.index, values=activity_dist.values,
                marker_colors=colors[:len(activity_dist)]
            ), row=3, col=3)

        fig.update_layout(
            height=1000,
            title_text='<b>Social Media Trend Analysis Dashboard</b>',
            title_x=0.5,
            title_font_size=22,
            showlegend=False
        )

        if save:
            fig.write_html(self.output_dir / 'trend_dashboard.html')
            fig.write_image(self.output_dir / 'trend_dashboard.png')

        return fig
