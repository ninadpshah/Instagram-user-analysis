# Instagram User Analysis

Analysis of Instagram user behaviour, engagement, and audience segmentation, with the other major social platforms included as comparison baselines. Covers EDA, K-means segmentation into five user personas, posting-pattern trend analysis, and an interactive Plotly dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **On the data in this repo:** the checked-in `data/sample_social_media_data.csv` is **generated sample data**, not real user data — 100 rows written to match the schema of the Kaggle dataset so the notebook runs immediately after cloning. `data/generate_sample_data.py` produces a larger 5,000-row synthetic set with the same schema. Any figure below is a demonstration of the pipeline, not a finding about real Instagram users. See [Using the real dataset](#using-the-real-dataset) to swap in the Kaggle file.

## What it does

- **Exploratory analysis** — distributions, missing-value handling, and summary statistics across follower counts, engagement rates, and account age
- **Engagement modelling** — which account and content attributes move engagement rate
- **User segmentation** — K-means clustering over scaled behavioural features to produce interpretable personas
- **Trend analysis** — posting frequency, peak activity hour, and content-type performance
- **Dashboards** — interactive multi-panel Plotly views plus static matplotlib/seaborn charts

## Dataset

The intended source is the [Social Media User Analysis](https://www.kaggle.com/datasets/rockyt07/social-media-user-analysis) dataset on Kaggle: ~5,000 user records across 6 platforms and 15 countries, with followers, following, post counts, likes/comments/shares, engagement rate, verification status, content type, posting frequency, peak activity hour, and interests.

Instagram is the largest platform slice and the focus of the analysis; Twitter, TikTok, YouTube, LinkedIn, and Facebook are retained as comparison groups.

## Project structure

```
Instagram-user-analysis/
├── data/
│   ├── generate_sample_data.py       # synthetic data generator (5,000 rows)
│   └── sample_social_media_data.csv  # 100-row sample, checked in so the repo runs as-is
├── notebooks/
│   └── social_media_analysis.ipynb   # main analysis
├── src/
│   ├── data_loader.py                # loading, cleaning, derived columns
│   ├── visualizations.py             # chart and dashboard builders
│   ├── user_segmentation.py          # K-means clustering and persona labelling
│   └── trend_analysis.py             # posting and content-performance trends
├── outputs/
│   ├── visualizations/               # generated charts (gitignored)
│   └── reports/                      # generated reports (gitignored)
├── requirements.txt
└── README.md
```
