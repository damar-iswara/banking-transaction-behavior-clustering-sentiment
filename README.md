# Banking-Transaction-Behavior-Clustering-Sentiment
A comprehensive dataset and analysis project focusing on banking transaction behaviors for clustering and sentiment-based classification. Includes data preprocessing, feature engineering, and machine learning models to uncover customer transaction patterns and sentiment insights.

This project is the final submission for the Clustering module in the Belajar Machine Learning Pemula (BMLP) course by Dicoding.
The notebook applies unsupervised learning using the K-Means algorithm to group transaction data based on user behavior patterns.
The main objective is to discover meaningful insights and detect potential anomalies that could be useful for fraud detection and customer segmentation.

*Import Libraries & Dataset Initialization*
```python
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
```
- **pandas, numpy**: data handling and numerical computation.
- **seaborn, matplotlib**: for visualization during EDA and clustering analysis.
- **KMeans, PCA**: clustering algorithm and dimensionality reduction.
- **MinMaxScaler, LabelEncoder**: preprocessing tools to normalize and encode data.
- **Silhouette Score & KElbowVisualizer**: model evaluation metrics and visualization.

*Exploratory Data Analysis (EDA)*
EDA is performed to understand the relationships between features and identify which attributes are most relevant for clustering.
- The features customerage and accountbalance show the strongest positive correlation of 0.321.
- These two features were selected as the primary variables for the clustering process.
<img width="1099" height="834" alt="download" src="https://github.com/user-attachments/assets/bede7e51-69b6-414a-9904-419b8c134349" />

*Data Preprocessing*
The preprocessing phase ensures the dataset is clean and ready for clustering.
1. Normalization
   Applied MinMaxScaler to rescale all numeric features into a uniform range of [0, 1].
   ```python
    scaler = MinMaxScaler()
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[num_columns] = scaler.fit_transform(df[num_columns])
   ```
3. Categorical Encoding
   - Identified non-date object columns.
   - Replaced empty strings and missing values with the mode.
   - Encoded categorical data using LabelEncoder.
    ```python
      encoder = LabelEncoder()
      df[col] = encoder.fit_transform(df[col])
     ```
4. Missing Values & Duplicates Handling
   - Filled missing numeric values with mean.
   - Filled missing categorical values with mode.
   - Removed duplicate records using df.drop_duplicates(inplace=True).
