"""
Visualization Module for Social Media User Analysis
Professional charts and graphs for portfolio presentation
Author: Data Analytics Portfolio Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color palettes
BRAND_COLORS = {
    'Instagram': '#E4405F',
    'Twitter': '#1DA1F2',
    'TikTok': '#000000',
    'YouTube': '#FF0000',
    'LinkedIn': '#0A66C2',
    'Facebook': '#1877F2'
}

PALETTE_MAIN = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
PALETTE_GRADIENT = ['#667eea', '#764ba2', '#6B8DD6', '#8E37D7', '#B721FF']


class SocialMediaVisualizer:
    """
    A class for creating professional visualizations for social media analysis.
    Supports both static (matplotlib/seaborn) and interactive (plotly) charts.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    def plot_platform_distribution(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create an interactive donut chart showing user distribution by platform.
        """
        platform_counts = df['platform'].value_counts()

        colors = [BRAND_COLORS.get(p, '#667eea') for p in platform_counts.index]

        fig = go.Figure(data=[go.Pie(
            labels=platform_counts.index,
            values=platform_counts.values,
            hole=0.5,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside',
            pull=[0.02] * len(platform_counts)
        )])

        fig.update_layout(
            title={
                'text': 'User Distribution by Social Media Platform',
                'x': 0.5,
                'font': {'size': 20}
            },
            showlegend=True,
            legend={'orientation': 'h', 'y': -0.1},
            annotations=[{
                'text': f'{len(df):,}<br>Users',
                'x': 0.5, 'y': 0.5,
                'font_size': 20,
                'showarrow': False
            }]
        )

        if save:
            fig.write_html(self.output_dir / 'platform_distribution.html')
            fig.write_image(self.output_dir / 'platform_distribution.png')

        return fig

    def plot_engagement_analysis(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Create a comprehensive engagement analysis dashboard.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Social Media Engagement Analysis Dashboard', fontsize=18, fontweight='bold', y=1.02)

        # 1. Engagement Rate Distribution by Platform
        ax1 = axes[0, 0]
        platform_order = df.groupby('platform')['avg_engagement_rate'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x='platform', y='avg_engagement_rate', order=platform_order,
                   palette=[BRAND_COLORS.get(p, '#667eea') for p in platform_order], ax=ax1)
        ax1.set_title('Engagement Rate Distribution by Platform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Platform')
        ax1.set_ylabel('Average Engagement Rate (%)')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Followers vs Engagement Scatter
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['followers'], df['avg_engagement_rate'],
                             c=df['is_verified'].map({True: '#667eea', False: '#f5576c'}),
                             alpha=0.6, s=50)
        ax2.set_xscale('log')
        ax2.set_title('Followers vs Engagement Rate', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Followers (log scale)')
        ax2.set_ylabel('Engagement Rate (%)')

        # Add legend
        handles = [plt.scatter([], [], c='#667eea', label='Verified'),
                  plt.scatter([], [], c='#f5576c', label='Not Verified')]
        ax2.legend(handles=handles, loc='upper right')

        # 3. Content Type Performance
        ax3 = axes[1, 0]
        content_engagement = df.groupby('content_type')['avg_engagement_rate'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        bars = ax3.barh(content_engagement.index, content_engagement['mean'],
                       color=PALETTE_MAIN[:len(content_engagement)], edgecolor='white', linewidth=1)
        ax3.errorbar(content_engagement['mean'], content_engagement.index,
                    xerr=content_engagement['std'], fmt='none', color='gray', capsize=3)
        ax3.set_title('Average Engagement by Content Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average Engagement Rate (%)')

        # Add value labels
        for bar, val in zip(bars, content_engagement['mean']):
            ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', fontsize=10)

        # 4. Engagement by Age Group
        ax4 = axes[1, 1]
        if 'age_group' in df.columns:
            age_engagement = df.groupby('age_group')['avg_engagement_rate'].mean().sort_index()
            ax4.bar(age_engagement.index.astype(str), age_engagement.values,
                   color=PALETTE_GRADIENT[:len(age_engagement)], edgecolor='white', linewidth=1)
            ax4.set_title('Engagement Rate by Age Group', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Age Group')
            ax4.set_ylabel('Average Engagement Rate (%)')
            ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / 'engagement_analysis.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_follower_growth_analysis(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create interactive visualization for follower analysis.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Follower Distribution by Category',
                'Followers by Platform',
                'Verified vs Non-Verified Users',
                'Top 10 Users by Followers'
            ),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # 1. Follower Category Pie
        if 'follower_category' in df.columns:
            cat_counts = df['follower_category'].value_counts()
            fig.add_trace(
                go.Pie(labels=cat_counts.index, values=cat_counts.values,
                      marker_colors=PALETTE_MAIN, hole=0.3),
                row=1, col=1
            )

        # 2. Average Followers by Platform
        platform_followers = df.groupby('platform')['followers'].mean().sort_values(ascending=True)
        colors = [BRAND_COLORS.get(p, '#667eea') for p in platform_followers.index]
        fig.add_trace(
            go.Bar(y=platform_followers.index, x=platform_followers.values,
                  orientation='h', marker_color=colors),
            row=1, col=2
        )

        # 3. Verified vs Non-Verified
        verified_data = df.groupby('is_verified')['followers'].agg(['mean', 'count'])
        verified_data.index = verified_data.index.map({True: 'Verified', False: 'Not Verified'})
        fig.add_trace(
            go.Bar(x=verified_data.index, y=verified_data['mean'],
                  marker_color=['#667eea', '#f5576c'],
                  text=[f'{v:,.0f}' for v in verified_data['mean']],
                  textposition='outside'),
            row=2, col=1
        )

        # 4. Top 10 Users
        top_users = df.nlargest(10, 'followers')[['username', 'followers', 'platform']]
        colors = [BRAND_COLORS.get(p, '#667eea') for p in top_users['platform']]
        fig.add_trace(
            go.Bar(x=top_users['username'], y=top_users['followers'],
                  marker_color=colors,
                  text=[f'{v:,.0f}' for v in top_users['followers']],
                  textposition='outside'),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text='<b>Follower Growth & Distribution Analysis</b>',
            showlegend=False,
            title_x=0.5
        )

        if save:
            fig.write_html(self.output_dir / 'follower_analysis.html')
            fig.write_image(self.output_dir / 'follower_analysis.png')

        return fig
