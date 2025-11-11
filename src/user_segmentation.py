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
