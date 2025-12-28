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

    def plot_geographic_distribution(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create a geographic analysis of user distribution.
        """
        country_data = df.groupby('country').agg({
            'user_id': 'count',
            'followers': 'sum',
            'avg_engagement_rate': 'mean'
        }).rename(columns={'user_id': 'user_count'}).reset_index()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Users by Country', 'Engagement Rate by Country'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Users by Country
        sorted_by_users = country_data.sort_values('user_count', ascending=True).tail(15)
        fig.add_trace(
            go.Bar(y=sorted_by_users['country'], x=sorted_by_users['user_count'],
                  orientation='h', marker_color='#667eea',
                  text=sorted_by_users['user_count'], textposition='outside'),
            row=1, col=1
        )

        # Engagement by Country
        sorted_by_engagement = country_data.sort_values('avg_engagement_rate', ascending=True).tail(15)
        fig.add_trace(
            go.Bar(y=sorted_by_engagement['country'], x=sorted_by_engagement['avg_engagement_rate'],
                  orientation='h', marker_color='#764ba2',
                  text=[f'{v:.1f}%' for v in sorted_by_engagement['avg_engagement_rate']],
                  textposition='outside'),
            row=1, col=2
        )

        fig.update_layout(
            height=600,
            title_text='<b>Geographic Distribution Analysis</b>',
            showlegend=False,
            title_x=0.5
        )

        if save:
            fig.write_html(self.output_dir / 'geographic_analysis.html')
            fig.write_image(self.output_dir / 'geographic_analysis.png')

        return fig

    def plot_activity_patterns(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Analyze and visualize posting activity patterns.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('User Activity & Posting Patterns', fontsize=16, fontweight='bold', y=1.05)

        # 1. Peak Activity Hours Heatmap
        ax1 = axes[0]
        hour_platform = df.pivot_table(
            values='user_id',
            index='platform',
            columns='peak_activity_hour',
            aggfunc='count'
        ).fillna(0)
        sns.heatmap(hour_platform, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'User Count'})
        ax1.set_title('Peak Activity Hours by Platform', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Platform')

        # 2. Posting Frequency Distribution
        ax2 = axes[1]
        freq_order = ['Multiple Daily', 'Daily', 'Weekly', 'Monthly']
        freq_counts = df['posting_frequency'].value_counts()
        # Reorder based on freq_order
        freq_counts = freq_counts.reindex([f for f in freq_order if f in freq_counts.index])

        bars = ax2.bar(freq_counts.index, freq_counts.values, color=PALETTE_MAIN[:len(freq_counts)],
                      edgecolor='white', linewidth=2)
        ax2.set_title('Posting Frequency Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Posting Frequency')
        ax2.set_ylabel('Number of Users')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 3. Posts vs Engagement
        ax3 = axes[2]
        sc = ax3.scatter(df['posts'], df['total_engagement'] if 'total_engagement' in df.columns else df['likes_received'],
                        c=df['followers'], cmap='viridis', alpha=0.6, s=50)
        ax3.set_xlabel('Number of Posts')
        ax3.set_ylabel('Total Engagement')
        ax3.set_title('Posts vs Engagement (colored by followers)', fontsize=12, fontweight='bold')
        plt.colorbar(sc, ax=ax3, label='Followers')

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / 'activity_patterns.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Create a correlation heatmap for numeric variables.
        """
        numeric_cols = ['followers', 'following', 'posts', 'likes_received',
                       'comments_received', 'shares_received', 'avg_engagement_rate',
                       'age', 'peak_activity_hour']
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdYlBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   cbar_kws={'label': 'Correlation Coefficient'})

        ax.set_title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_interests_wordcloud(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Create a word cloud of user interests.
        """
        try:
            from wordcloud import WordCloud

            # Combine all interests
            all_interests = '|'.join(df['interests'].dropna())
            interests_list = all_interests.replace('|', ' ')

            wordcloud = WordCloud(
                width=1200, height=600,
                background_color='white',
                colormap='viridis',
                max_words=100,
                min_font_size=10
            ).generate(interests_list)

            fig, ax = plt.subplots(figsize=(14, 7))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('User Interests Word Cloud', fontsize=18, fontweight='bold', pad=20)

            if save:
                fig.savefig(self.output_dir / 'interests_wordcloud.png', dpi=300, bbox_inches='tight')

            return fig

        except ImportError:
            print("WordCloud library not installed. Skipping word cloud visualization.")
            return None

    def create_executive_summary(self, df: pd.DataFrame, stats: dict, save: bool = True) -> go.Figure:
        """
        Create an executive summary dashboard with key metrics.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '', '', '',
                'Platform Share', 'Engagement by Content Type', 'Age Distribution',
                'Follower Categories', 'Top Interests', 'Activity Levels'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'pie'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        # Key Metrics (Row 1)
        fig.add_trace(go.Indicator(
            mode="number",
            value=stats.get('total_users', 0),
            title={"text": "Total Users"},
            number={'font': {'size': 40, 'color': '#667eea'}}
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="number",
            value=stats.get('total_followers', 0),
            title={"text": "Total Followers"},
            number={'font': {'size': 40, 'color': '#764ba2'}, 'valueformat': ',.0f'}
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="number",
            value=stats.get('avg_engagement_rate', 0),
            title={"text": "Avg Engagement Rate"},
            number={'font': {'size': 40, 'color': '#f5576c'}, 'suffix': '%', 'valueformat': '.1f'}
        ), row=1, col=3)

        # Platform Share (Row 2)
        platform_counts = df['platform'].value_counts()
        colors = [BRAND_COLORS.get(p, '#667eea') for p in platform_counts.index]
        fig.add_trace(go.Pie(
            labels=platform_counts.index, values=platform_counts.values,
            marker_colors=colors, hole=0.4
        ), row=2, col=1)

        # Engagement by Content Type (Row 2)
        content_eng = df.groupby('content_type')['avg_engagement_rate'].mean().sort_values()
        fig.add_trace(go.Bar(
            y=content_eng.index, x=content_eng.values,
            orientation='h', marker_color=PALETTE_MAIN[:len(content_eng)]
        ), row=2, col=2)

        # Age Distribution (Row 2)
        if 'age_group' in df.columns:
            age_counts = df['age_group'].value_counts().sort_index()
            fig.add_trace(go.Bar(
                x=age_counts.index.astype(str), y=age_counts.values,
                marker_color=PALETTE_GRADIENT[:len(age_counts)]
            ), row=2, col=3)

        # Follower Categories (Row 3)
        if 'follower_category' in df.columns:
            cat_counts = df['follower_category'].value_counts()
            fig.add_trace(go.Pie(
                labels=cat_counts.index, values=cat_counts.values,
                marker_colors=PALETTE_MAIN[:len(cat_counts)], hole=0.3
            ), row=3, col=1)

        # Top Interests (Row 3)
        if 'primary_interest' in df.columns:
            interest_counts = df['primary_interest'].value_counts().head(8)
            fig.add_trace(go.Bar(
                y=interest_counts.index, x=interest_counts.values,
                orientation='h', marker_color='#667eea'
            ), row=3, col=2)

        # Activity Levels (Row 3)
        if 'activity_level' in df.columns:
            activity_counts = df['activity_level'].value_counts()
            fig.add_trace(go.Pie(
                labels=activity_counts.index, values=activity_counts.values,
                marker_colors=['#00f2fe', '#4facfe', '#667eea', '#764ba2'][:len(activity_counts)]
            ), row=3, col=3)

        fig.update_layout(
            height=1000,
            title_text='<b>Social Media User Analysis - Executive Summary</b>',
            title_x=0.5,
            title_font_size=24,
            showlegend=False
        )

        if save:
            fig.write_html(self.output_dir / 'executive_summary.html')
            fig.write_image(self.output_dir / 'executive_summary.png')

        return fig

    def generate_all_visualizations(self, df: pd.DataFrame, stats: dict) -> None:
        """
        Generate all visualizations and save them to the output directory.
        """
        print("\nGenerating visualizations...")
        print("-" * 50)

        self.plot_platform_distribution(df)
        print("✓ Platform distribution chart created")

        self.plot_engagement_analysis(df)
        print("✓ Engagement analysis dashboard created")

        self.plot_follower_growth_analysis(df)
        print("✓ Follower growth analysis created")

        self.plot_geographic_distribution(df)
        print("✓ Geographic distribution chart created")

        self.plot_activity_patterns(df)
        print("✓ Activity patterns chart created")

        self.plot_correlation_matrix(df)
        print("✓ Correlation matrix created")

        self.plot_interests_wordcloud(df)
        print("✓ Interests word cloud created")

        self.create_executive_summary(df, stats)
        print("✓ Executive summary dashboard created")

        print("-" * 50)
        print(f"All visualizations saved to: {self.output_dir}")


if __name__ == "__main__":
    # Test with sample data
    from data_loader import SocialMediaDataLoader

    loader = SocialMediaDataLoader()
    df, stats = loader.prepare_data()

    visualizer = SocialMediaVisualizer()
    visualizer.generate_all_visualizations(df, stats)
