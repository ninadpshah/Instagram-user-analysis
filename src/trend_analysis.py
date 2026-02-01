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

    def plot_engagement_insights(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Create detailed engagement insights visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Engagement Pattern Insights', fontsize=18, fontweight='bold', y=1.02)

        # 1. Engagement vs Followers (Log scale)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            df['followers'], df['avg_engagement_rate'],
            c=df['posts'], cmap='viridis', alpha=0.5, s=30
        )
        ax1.set_xscale('log')
        ax1.set_xlabel('Followers (log scale)', fontsize=12)
        ax1.set_ylabel('Engagement Rate (%)', fontsize=12)
        ax1.set_title('The Engagement Paradox: Followers vs Engagement', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Number of Posts')

        # Add trend line
        log_followers = np.log10(df['followers'] + 1)
        z = np.polyfit(log_followers, df['avg_engagement_rate'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(log_followers.min(), log_followers.max(), 100)
        ax1.plot(10**x_line, p(x_line), 'r--', linewidth=2, label='Trend line')
        ax1.legend()

        # 2. Engagement by Hour Heatmap
        ax2 = axes[0, 1]
        hour_platform = df.pivot_table(
            values='avg_engagement_rate',
            index='platform',
            columns='peak_activity_hour',
            aggfunc='mean'
        )
        sns.heatmap(hour_platform, cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Avg Engagement %'})
        ax2.set_title('Engagement by Platform and Hour', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Platform')

        # 3. Content Performance Comparison
        ax3 = axes[1, 0]
        content_metrics = df.groupby('content_type').agg({
            'avg_engagement_rate': 'mean',
            'likes_received': 'mean',
            'comments_received': 'mean'
        })
        content_metrics_normalized = content_metrics / content_metrics.max()
        content_metrics_normalized.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Content Type Performance (Normalized)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Content Type')
        ax3.set_ylabel('Normalized Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Metric')

        # 4. Top Performers Analysis
        ax4 = axes[1, 1]
        qualified = df[df['followers'] >= 1000]
        top_engaged = qualified.nlargest(20, 'avg_engagement_rate')

        colors_list = ['#667eea' if v else '#f5576c' for v in top_engaged['is_verified']]
        bars = ax4.barh(range(len(top_engaged)), top_engaged['avg_engagement_rate'], color=colors_list)
        ax4.set_yticks(range(len(top_engaged)))
        ax4.set_yticklabels([f"{u[:15]}..." if len(u) > 15 else u for u in top_engaged['username']])
        ax4.set_xlabel('Engagement Rate (%)')
        ax4.set_title('Top 20 Most Engaging Users (≥1K followers)', fontsize=14, fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#667eea', label='Verified'),
                         Patch(facecolor='#f5576c', label='Not Verified')]
        ax4.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / 'engagement_insights.png', dpi=300, bbox_inches='tight')

        return fig

    def generate_insights_report(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive insights report.
        """
        # Run all analyses
        self.insights = []
        self.analyze_engagement_trends(df)
        self.analyze_platform_trends(df)
        self.analyze_demographic_trends(df)
        self.analyze_activity_patterns(df)
        self.analyze_verified_vs_unverified(df)

        report = []
        report.append("=" * 70)
        report.append("SOCIAL MEDIA USER ANALYSIS - KEY INSIGHTS REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Users Analyzed: {len(df):,}")
        report.append(f"Platforms Covered: {df['platform'].nunique()}")
        report.append(f"Countries Represented: {df['country'].nunique()}")

        report.append("\n" + "=" * 70)
        report.append("KEY FINDINGS")
        report.append("=" * 70)

        for i, insight in enumerate(self.insights, 1):
            report.append(f"\n{i}. {insight}")

        # Additional computed insights
        report.append("\n" + "=" * 70)
        report.append("STATISTICAL SUMMARY")
        report.append("=" * 70)

        report.append(f"\nEngagement Metrics:")
        report.append(f"  - Average Engagement Rate: {df['avg_engagement_rate'].mean():.2f}%")
        report.append(f"  - Median Engagement Rate: {df['avg_engagement_rate'].median():.2f}%")
        report.append(f"  - Highest Engagement Platform: {df.groupby('platform')['avg_engagement_rate'].mean().idxmax()}")

        report.append(f"\nFollower Statistics:")
        report.append(f"  - Average Followers: {df['followers'].mean():,.0f}")
        report.append(f"  - Median Followers: {df['followers'].median():,.0f}")
        report.append(f"  - Total Reach: {df['followers'].sum():,}")

        report.append(f"\nContent Insights:")
        report.append(f"  - Most Popular Content: {df['content_type'].value_counts().idxmax()}")
        report.append(f"  - Highest Engaging Content: {df.groupby('content_type')['avg_engagement_rate'].mean().idxmax()}")

        report.append(f"\nUser Activity:")
        report.append(f"  - Verified Users: {df['is_verified'].sum()} ({df['is_verified'].mean()*100:.1f}%)")
        report.append(f"  - Most Active Age Group: {df.groupby('age_group')['posts'].mean().idxmax() if 'age_group' in df.columns else 'N/A'}")

        report.append("\n" + "=" * 70)
        report.append("RECOMMENDATIONS")
        report.append("=" * 70)

        # Generate recommendations based on data
        best_content = df.groupby('content_type')['avg_engagement_rate'].mean().idxmax()
        best_time = df.groupby('peak_activity_hour')['avg_engagement_rate'].mean().idxmax()
        best_platform = df.groupby('platform')['avg_engagement_rate'].mean().idxmax()

        report.append(f"\n1. Content Strategy: Focus on {best_content} content for maximum engagement")
        report.append(f"2. Posting Time: Optimal posting time is around {best_time}:00")
        report.append(f"3. Platform Focus: {best_platform} shows highest average engagement")
        report.append(f"4. Engagement Focus: Micro-influencers show higher engagement rates than mega-influencers")

        report_text = "\n".join(report)

        # Save report
        reports_dir = self.output_dir.parent / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / 'insights_report.txt', 'w') as f:
            f.write(report_text)

        return report_text

    def run_full_analysis(self, df: pd.DataFrame) -> Tuple[Dict, str]:
        """
        Run complete trend analysis pipeline.
        """
        print("\n" + "=" * 50)
        print("RUNNING TREND ANALYSIS")
        print("=" * 50)

        all_trends = {}
        all_trends['engagement'] = self.analyze_engagement_trends(df)
        all_trends['platform'] = self.analyze_platform_trends(df)
        all_trends['demographic'] = self.analyze_demographic_trends(df)
        all_trends['activity'] = self.analyze_activity_patterns(df)
        all_trends['verification'] = self.analyze_verified_vs_unverified(df)
        all_trends['top_performers'] = self.identify_top_performers(df)

        # Generate visualizations
        self.plot_trend_dashboard(df)
        print("✓ Trend dashboard created")

        self.plot_engagement_insights(df)
        print("✓ Engagement insights visualization created")

        # Generate report
        report = self.generate_insights_report(df)

        print("\n" + "=" * 50)
        print("TREND ANALYSIS COMPLETE")
        print("=" * 50)

        return all_trends, report


if __name__ == "__main__":
    from data_loader import SocialMediaDataLoader

    # Load and prepare data
    loader = SocialMediaDataLoader()
    df, stats = loader.prepare_data()

    # Run trend analysis
    analyzer = TrendAnalyzer()
    trends, report = analyzer.run_full_analysis(df)

    print(report)
