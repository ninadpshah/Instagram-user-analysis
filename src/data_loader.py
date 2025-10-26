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
