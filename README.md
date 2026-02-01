# Social Media User Analysis

A comprehensive data analytics portfolio project analyzing social media user behavior, engagement patterns, and trends across multiple platforms.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview

This project performs in-depth analysis of social media user data to uncover actionable insights about user behavior, engagement patterns, and platform-specific trends. The analysis covers users across Instagram, Twitter, TikTok, YouTube, LinkedIn, and Facebook.

### Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis and data visualization
- **Engagement Analysis**: Understanding factors that drive user engagement
- **User Segmentation**: K-means clustering to identify distinct user personas
- **Trend Analysis**: Identifying patterns in posting behavior and content performance
- **Interactive Dashboards**: Professional visualizations using Plotly

## Dataset

This project uses the [Social Media User Analysis](https://www.kaggle.com/datasets/rockyt07/social-media-user-analysis) dataset from Kaggle, which contains:

- **5,000+** user records
- **6** social media platforms
- **15+** countries represented
- Features including followers, engagement rates, content types, posting frequency, and more

## Project Structure

```
social-media-analysis/
│
├── data/
│   └── social_media_users.csv       # Raw dataset (download from Kaggle)
│
├── notebooks/
│   └── social_media_analysis.ipynb  # Main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── visualizations.py            # Visualization functions
│   ├── user_segmentation.py         # Clustering analysis
│   └── trend_analysis.py            # Trend identification
│
├── outputs/
│   ├── visualizations/              # Generated charts and dashboards
│   └── reports/                     # Analysis reports
│
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/social-media-analysis.git
cd social-media-analysis
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/rockyt07/social-media-user-analysis)
   - Download and place `social_media_users.csv` in the `data/` folder

## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook notebooks/social_media_analysis.ipynb
```

### Running Individual Modules

```python
# Data Loading
from src.data_loader import SocialMediaDataLoader
loader = SocialMediaDataLoader('data/social_media_users.csv')
df, stats = loader.prepare_data()

# Visualizations
from src.visualizations import SocialMediaVisualizer
viz = SocialMediaVisualizer('outputs/visualizations')
viz.generate_all_visualizations(df, stats)

# User Segmentation
from src.user_segmentation import UserSegmentation
seg = UserSegmentation()
df, report = seg.run_full_analysis(df, n_clusters=5)

# Trend Analysis
from src.trend_analysis import TrendAnalyzer
analyzer = TrendAnalyzer()
trends, insights = analyzer.run_full_analysis(df)
```

## Key Findings

### 1. The Engagement Paradox
Users with fewer followers (micro-influencers) consistently show **higher engagement rates** than mega-influencers, suggesting quality over quantity in audience building.

### 2. Platform Performance
- **TikTok** and **Instagram** lead in average engagement rates
- **LinkedIn** shows highest engagement for professional/business content
- **YouTube** users have the largest average follower counts

### 3. Content Strategy Insights
- **Video content** generates 2-3x more engagement than static images
- **Reels/Short-form video** outperforms all other content types
- Posting during **evening hours (17:00-21:00)** maximizes engagement

### 4. User Segments Identified
| Segment | Characteristics | Engagement |
|---------|-----------------|------------|
| Celebrity Influencers | 100K+ followers, verified | Low |
| Growing Influencers | 10K-100K followers | Medium |
| Micro-Influencers | 1K-10K followers | High |
| Engaged Newcomers | <1K followers | Very High |
| Casual Users | Sporadic posting | Variable |

## Visualizations

The project generates various professional visualizations:

- **Executive Summary Dashboard**: Key metrics at a glance
- **Platform Distribution Chart**: User breakdown by platform
- **Engagement Analysis Dashboard**: Multi-panel engagement insights
- **Cluster Analysis**: User segmentation visualizations
- **Trend Dashboard**: Activity patterns and trends
- **Correlation Matrix**: Feature relationships
- **Geographic Distribution**: User location analysis

## Technologies Used

- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning (K-means clustering)
- **Jupyter**: Interactive notebook environment

## Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Visualization (Static & Interactive)
- Machine Learning (Unsupervised Learning)
- Feature Engineering
- Business Insights Generation
- Python Programming
- Documentation & Reporting

## Future Improvements

- [ ] Add sentiment analysis of user bios/posts
- [ ] Implement time-series forecasting for follower growth
- [ ] Build a recommendation system for content strategy
- [ ] Create a Streamlit dashboard for interactive exploration
- [ ] Add A/B testing analysis capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Data Analytics Portfolio Project**

---

*This project was created as part of a data analytics portfolio to demonstrate proficiency in data analysis, visualization, and machine learning techniques.*
