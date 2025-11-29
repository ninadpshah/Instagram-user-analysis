"""
User Segmentation Module for Social Media User Analysis
Implements clustering and segmentation analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class UserSegmentation:
    """
    A class for performing user segmentation and clustering analysis.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the segmentation module.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.cluster_profiles = None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for clustering analysis.
        """
        feature_cols = [
            'followers', 'following', 'posts', 'avg_engagement_rate',
            'likes_received', 'comments_received', 'shares_received',
            'account_age_days', 'follow_ratio', 'posts_per_day',
            'engagement_per_post', 'peak_activity_hour'
        ]

        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Create feature matrix
        X = df[feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Log transform skewed features
        skewed_cols = ['followers', 'following', 'posts', 'likes_received',
                      'comments_received', 'shares_received', 'engagement_per_post']
        for col in skewed_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        return X, feature_cols

    def find_optimal_clusters(self, X: pd.DataFrame, max_k: int = 10) -> Tuple[int, plt.Figure]:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        """
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # Find optimal k using silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow plot
        axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Silhouette score plot
        axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score for Optimal k', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / 'optimal_clusters.png', dpi=300, bbox_inches='tight')

        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.3f}")

        return optimal_k, fig

    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform K-means clustering on the user data.
        """
        X, feature_cols = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        # Fit KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = self.kmeans.fit_predict(X_scaled)

        # Perform PCA for visualization
        self.pca = PCA(n_components=2)
        pca_result = self.pca.fit_transform(X_scaled)
        df['pca_1'] = pca_result[:, 0]
        df['pca_2'] = pca_result[:, 1]

        print(f"Clustering complete. Silhouette score: {silhouette_score(X_scaled, df['cluster']):.3f}")

        return df

    def analyze_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze and profile each cluster.
        """
        analysis_cols = [
            'followers', 'following', 'posts', 'avg_engagement_rate',
            'total_engagement', 'account_age_days', 'follow_ratio',
            'posts_per_day', 'engagement_per_post'
        ]
        analysis_cols = [col for col in analysis_cols if col in df.columns]

        # Calculate cluster profiles
        self.cluster_profiles = df.groupby('cluster')[analysis_cols].agg(['mean', 'median', 'std'])

        # Calculate cluster sizes
        cluster_sizes = df['cluster'].value_counts().sort_index()

        # Create summary
        summary = df.groupby('cluster').agg({
            'followers': 'mean',
            'avg_engagement_rate': 'mean',
            'posts': 'mean',
            'is_verified': 'mean',
            'user_id': 'count'
        }).round(2)

        summary.columns = ['Avg Followers', 'Avg Engagement %', 'Avg Posts', '% Verified', 'User Count']
        summary['% Verified'] = (summary['% Verified'] * 100).round(1)

        # Assign cluster names based on characteristics
        cluster_names = self._assign_cluster_names(summary)
        df['cluster_name'] = df['cluster'].map(cluster_names)

        return summary

    def _assign_cluster_names(self, summary: pd.DataFrame) -> dict:
        """
        Assign meaningful names to clusters based on their characteristics.
        """
        names = {}

        for cluster_id in summary.index:
            row = summary.loc[cluster_id]
            followers = row['Avg Followers']
            engagement = row['Avg Engagement %']

            if followers > 100000:
                if engagement > 5:
                    names[cluster_id] = 'Celebrity Influencers'
                else:
                    names[cluster_id] = 'Mass Reach Accounts'
            elif followers > 10000:
                if engagement > 5:
                    names[cluster_id] = 'Engaged Mid-tier'
                else:
                    names[cluster_id] = 'Growing Influencers'
            elif followers > 1000:
                if engagement > 5:
                    names[cluster_id] = 'Micro-Influencers'
                else:
                    names[cluster_id] = 'Active Community'
            else:
                if engagement > 5:
                    names[cluster_id] = 'Engaged Newcomers'
                else:
                    names[cluster_id] = 'Casual Users'

        return names

    def plot_cluster_visualization(self, df: pd.DataFrame, save: bool = True) -> go.Figure:
        """
        Create interactive cluster visualization.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'User Clusters (PCA Projection)',
                'Cluster Size Distribution',
                'Followers vs Engagement by Cluster',
                'Cluster Characteristics Radar'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'scatterpolar'}]
            ]
        )

        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b']

        # 1. PCA Scatter Plot
        for i, cluster in enumerate(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            cluster_name = cluster_data['cluster_name'].iloc[0] if 'cluster_name' in df.columns else f'Cluster {cluster}'
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['pca_1'],
                    y=cluster_data['pca_2'],
                    mode='markers',
                    name=cluster_name,
                    marker=dict(color=colors[i % len(colors)], size=6, opacity=0.6),
                    hovertemplate=f'{cluster_name}<br>Followers: %{{customdata[0]:,.0f}}<br>Engagement: %{{customdata[1]:.1f}}%',
                    customdata=cluster_data[['followers', 'avg_engagement_rate']].values
                ),
                row=1, col=1
            )

        # 2. Cluster Size Pie
        cluster_counts = df['cluster'].value_counts().sort_index()
        cluster_labels = [df[df['cluster'] == c]['cluster_name'].iloc[0] if 'cluster_name' in df.columns else f'Cluster {c}'
                        for c in cluster_counts.index]
        fig.add_trace(
            go.Pie(
                labels=cluster_labels,
                values=cluster_counts.values,
                marker_colors=colors[:len(cluster_counts)],
                hole=0.4
            ),
            row=1, col=2
        )

        # 3. Followers vs Engagement Scatter
        for i, cluster in enumerate(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster].sample(min(500, len(df[df['cluster'] == cluster])))
            cluster_name = cluster_data['cluster_name'].iloc[0] if 'cluster_name' in df.columns else f'Cluster {cluster}'
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['followers'],
                    y=cluster_data['avg_engagement_rate'],
                    mode='markers',
                    name=cluster_name,
                    marker=dict(color=colors[i % len(colors)], size=5, opacity=0.5),
                    showlegend=False
                ),
                row=2, col=1
            )

        # 4. Radar Chart for cluster characteristics
        if self.cluster_profiles is not None:
            metrics = ['followers', 'avg_engagement_rate', 'posts', 'follow_ratio']
            metrics = [m for m in metrics if m in df.columns]

            cluster_means = df.groupby('cluster')[metrics].mean()
            # Normalize for radar chart
            cluster_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

            for i, cluster in enumerate(cluster_normalized.index):
                cluster_name = df[df['cluster'] == cluster]['cluster_name'].iloc[0] if 'cluster_name' in df.columns else f'Cluster {cluster}'
                values = cluster_normalized.loc[cluster].tolist()
                values.append(values[0])  # Close the radar

                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=metrics + [metrics[0]],
                        fill='toself',
                        name=cluster_name,
                        line_color=colors[i % len(colors)],
                        opacity=0.6,
                        showlegend=False
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            height=900,
            title_text='<b>User Segmentation Analysis</b>',
            title_x=0.5,
            title_font_size=20
        )

        fig.update_xaxes(type='log', title_text='Followers (log)', row=2, col=1)
        fig.update_yaxes(title_text='Engagement Rate (%)', row=2, col=1)

        if save:
            fig.write_html(self.output_dir / 'cluster_analysis.html')
            fig.write_image(self.output_dir / 'cluster_analysis.png')

        return fig

    def plot_cluster_profiles(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Create detailed cluster profile visualizations.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Cluster Profiles', fontsize=18, fontweight='bold', y=1.02)

        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']

        # 1. Followers Distribution by Cluster
        ax1 = axes[0, 0]
        for i, cluster in enumerate(sorted(df['cluster'].unique())):
            cluster_data = df[df['cluster'] == cluster]['followers']
            ax1.hist(np.log10(cluster_data + 1), bins=30, alpha=0.5,
                    label=df[df['cluster'] == cluster]['cluster_name'].iloc[0] if 'cluster_name' in df.columns else f'Cluster {cluster}',
                    color=colors[i % len(colors)])
        ax1.set_xlabel('Log10(Followers)')
        ax1.set_ylabel('Count')
        ax1.set_title('Follower Distribution by Cluster', fontweight='bold')
        ax1.legend(fontsize=8)

        # 2. Engagement Rate by Cluster
        ax2 = axes[0, 1]
        cluster_order = df.groupby('cluster')['avg_engagement_rate'].median().sort_values().index
        palette = {c: colors[i % len(colors)] for i, c in enumerate(cluster_order)}
        sns.boxplot(data=df, x='cluster', y='avg_engagement_rate', order=cluster_order,
                   palette=palette, ax=ax2)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Engagement Rate (%)')
        ax2.set_title('Engagement Rate by Cluster', fontweight='bold')

        # 3. Platform Distribution by Cluster
        ax3 = axes[0, 2]
        platform_cluster = pd.crosstab(df['platform'], df['cluster'], normalize='columns') * 100
        platform_cluster.plot(kind='bar', ax=ax3, color=colors[:len(df['cluster'].unique())])
        ax3.set_xlabel('Platform')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Platform Distribution by Cluster', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Cluster', fontsize=8)

        # 4. Content Type by Cluster
        ax4 = axes[1, 0]
        content_cluster = pd.crosstab(df['content_type'], df['cluster'], normalize='columns') * 100
        content_cluster.plot(kind='bar', ax=ax4, color=colors[:len(df['cluster'].unique())])
        ax4.set_xlabel('Content Type')
        ax4.set_ylabel('Percentage')
        ax4.set_title('Content Type by Cluster', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Cluster', fontsize=8)

        # 5. Age Distribution by Cluster
        ax5 = axes[1, 1]
        if 'age_group' in df.columns:
            age_cluster = pd.crosstab(df['age_group'], df['cluster'], normalize='columns') * 100
            age_cluster.plot(kind='bar', ax=ax5, color=colors[:len(df['cluster'].unique())])
            ax5.set_xlabel('Age Group')
            ax5.set_ylabel('Percentage')
            ax5.set_title('Age Distribution by Cluster', fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend(title='Cluster', fontsize=8)

        # 6. Activity Level by Cluster
        ax6 = axes[1, 2]
        if 'activity_level' in df.columns:
            activity_cluster = pd.crosstab(df['activity_level'], df['cluster'], normalize='columns') * 100
            activity_cluster.plot(kind='bar', ax=ax6, color=colors[:len(df['cluster'].unique())])
            ax6.set_xlabel('Activity Level')
            ax6.set_ylabel('Percentage')
            ax6.set_title('Activity Level by Cluster', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
            ax6.legend(title='Cluster', fontsize=8)

        plt.tight_layout()

        if save:
            fig.savefig(self.output_dir / 'cluster_profiles.png', dpi=300, bbox_inches='tight')

        return fig

    def generate_cluster_report(self, df: pd.DataFrame) -> str:
        """
        Generate a text report summarizing cluster characteristics.
        """
        report = []
        report.append("=" * 60)
        report.append("USER SEGMENTATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            cluster_name = cluster_data['cluster_name'].iloc[0] if 'cluster_name' in df.columns else f'Cluster {cluster}'

            report.append(f"\n{'='*40}")
            report.append(f"CLUSTER {cluster}: {cluster_name}")
            report.append(f"{'='*40}")
            report.append(f"Size: {len(cluster_data):,} users ({len(cluster_data)/len(df)*100:.1f}%)")
            report.append(f"\nKey Metrics:")
            report.append(f"  - Average Followers: {cluster_data['followers'].mean():,.0f}")
            report.append(f"  - Median Followers: {cluster_data['followers'].median():,.0f}")
            report.append(f"  - Average Engagement: {cluster_data['avg_engagement_rate'].mean():.2f}%")
            report.append(f"  - Average Posts: {cluster_data['posts'].mean():,.0f}")
            report.append(f"  - Verified Users: {cluster_data['is_verified'].sum()} ({cluster_data['is_verified'].mean()*100:.1f}%)")

            report.append(f"\nTop Platforms:")
            top_platforms = cluster_data['platform'].value_counts().head(3)
            for platform, count in top_platforms.items():
                report.append(f"  - {platform}: {count} ({count/len(cluster_data)*100:.1f}%)")

            report.append(f"\nTop Content Types:")
            top_content = cluster_data['content_type'].value_counts().head(3)
            for content, count in top_content.items():
                report.append(f"  - {content}: {count} ({count/len(cluster_data)*100:.1f}%)")

            if 'primary_interest' in df.columns:
                report.append(f"\nTop Interests:")
                top_interests = cluster_data['primary_interest'].value_counts().head(3)
                for interest, count in top_interests.items():
                    report.append(f"  - {interest}: {count} ({count/len(cluster_data)*100:.1f}%)")

        report_text = "\n".join(report)

        # Save report
        with open(self.output_dir.parent / 'reports' / 'cluster_report.txt', 'w') as f:
            f.write(report_text)

        return report_text

    def run_full_analysis(self, df: pd.DataFrame, n_clusters: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        Run complete segmentation analysis pipeline.
        """
        print("\n" + "="*50)
        print("RUNNING USER SEGMENTATION ANALYSIS")
        print("="*50)

        # Prepare features and find optimal clusters
        X, _ = self.prepare_features(df)
        optimal_k, _ = self.find_optimal_clusters(X)

        # Use provided n_clusters or optimal
        k = n_clusters if n_clusters else optimal_k

        # Perform clustering
        df = self.perform_clustering(df, n_clusters=k)

        # Analyze clusters
        summary = self.analyze_clusters(df)
        print("\nCluster Summary:")
        print(summary)

        # Generate visualizations
        self.plot_cluster_visualization(df)
        self.plot_cluster_profiles(df)

        # Generate report
        report = self.generate_cluster_report(df)

        print("\n" + "="*50)
        print("SEGMENTATION ANALYSIS COMPLETE")
        print("="*50)

        return df, report


if __name__ == "__main__":
    from data_loader import SocialMediaDataLoader

    # Load and prepare data
    loader = SocialMediaDataLoader()
    df, stats = loader.prepare_data()

    # Run segmentation analysis
    segmentation = UserSegmentation()
    df, report = segmentation.run_full_analysis(df, n_clusters=5)

    print(report)
