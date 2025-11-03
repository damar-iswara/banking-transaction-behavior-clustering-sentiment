# Banking Transaction Behavior Clustering & Sentiment

A comprehensive dataset and analysis project focusing on banking transaction behaviors for clustering and sentiment-based classification. Includes data preprocessing, feature engineering, and machine learning models to uncover customer transaction patterns and sentiment insights.

This project is the final submission for the Clustering module in the **Belajar Machine Learning Pemula (BMLP)** course by Dicoding. The notebook applies unsupervised learning using the **K-Means** algorithm to group transaction data based on user behavior patterns. The main objective is to discover meaningful insights and detect potential anomalies that could be useful for fraud detection and customer segmentation.

---

## 🛠️ Import Libraries & Dataset Initialization

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

* **pandas, numpy**: for data manipulation and numerical computation.
* **seaborn, matplotlib**: For data visualization during EDA and clustering analysis.
* **KMeans, PCA**: Clustering algorithm and dimensionality reduction techniques.
* **MinMaxScaler, LabelEncoder**: Preprocessing tools for data normalization and encoding.
* **Silhouette Score & KElbowVisualizer**: Metrics and visualization for model evaluation.

---

## 📊 Exploratory Data Analysis (EDA)

EDA is performed to understand the relationships between features and identify which attributes are most relevant for clustering.

* The features `customerage` and `accountbalance` show the strongest positive correlation of **0.321**.
* These two features were selected as the primary variables for the clustering process.

![Correlation Heatmap](https://github.com/user-attachments/assets/bede7e51-69b6-414a-9904-419b8c134349)

---

## ⚙️ Data Preprocessing

The preprocessing phase ensures the dataset is clean and ready for clustering.

1.  **Normalization**
    * Applied  `MinMaxScaler` to rescale all numeric features into a uniform range of [0, 1].
    ```python
    scaler = MinMaxScaler()
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[num_columns] = scaler.fit_transform(df[num_columns])
    ```

2.  **Categorical Encoding**
    * Identified non-date object columns.
    * Replaced empty strings and missing values with the mode (most frequent value).
    * Encoded categorical data using `LabelEncoder`.
    ```python
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    ```

3.  **Missing Values & Duplicates Handling**
    * MeFilled missing numeric values with the mean.
    * Filled missing categorical values with the mode.
    * Removed duplicate records using `df.drop_duplicates(inplace=True)`.

---

## 🤖 Clustering (K-Means)

The K-Means algorithm was applied to group the data based on customer age and account balance. The optimal number of clusters (k) was determined using the Elbow Method, while the Silhouette Score was used to evaluate clustering quality.

| Metric | Model | Value |
| :--- | :--- | :--- |
| Optimal k | K-Means | **3** |
| Silhouette Score | K-Means | **0.4887** |
| Silhouette Score | PCA-reduced | **0.438** |

* The Elbow Method indicates that **k=3** provides the most balanced grouping.
* A Silhouette Score of **0.4887** suggests moderately distinct clusters—the groups are separated enough to provide meaningful insights.

![Elbow Method for Optimal k](https://github.com/user-attachments/assets/1a3a8ec9-1496-42fa-9519-dd6070673963)

![Silhouette Plot](https://github.com/user-attachments/assets/74544b43-9f9a-44ed-9725-6a42a98f90d8)

---

## 💡 Cluster Interpretation & Business Insights

After determining the optimal number of clusters (k=3) and validating with the Silhouette Score (0.4887), each cluster was analyzed based on its inverse-transformed mean feature values to identify customer characteristics and potential business strategies.

### Cluster 1: Established Customers (High Balance & Transactions)

| Feature | Mean/Mode |
| :--- | :--- |
| TransactionAmount | 280.02 |
| CustomerAge | 44.70 |
| TransactionDuration | 121.31 |
| LoginAttempts | 1.11 |
| AccountBalance | 9983.68 |
| TransactionType | Debit |
| Channel | Branch |
| CustomerOccupation | Doctor |

This cluster represents affluent and professional customers with high account balances, large transaction amounts, and stable financial activity. Their dominant occupation as doctors indicates strong purchasing power and preference for branch-based transactions due to formal needs.

**Business Strategy:** Offer premium investment products, exclusive financial services, and personal banking solutions tailored for high-value customers.

### Cluster 2: Young Customers (Low Balance)

| Feature | Mean/Mode |
| :--- | :--- |
| TransactionAmount | 289.16 |
| CustomerAge | 26.32 |
| TransactionDuration | 119.78 |
| LoginAttempts | 1.12 |
| AccountBalance | 1874.32 |
| TransactionType | Debit |
| Channel | Branch |
| CustomerOccupation | Student |

This cluster consists mostly of young customers, primarily students or early-career individuals with limited savings but active transaction behavior. They represent a segment with long-term growth potential.

**Business Strategy:** Focus on financial education programs, student savings accounts, small cashback promotions, and digital engagement campaigns. Building early loyalty is key.

### Cluster 3: Senior Customers (Moderate Balance)

| Feature | Mean/Mode |
| :--- | :--- |
| TransactionAmount | 283.19 |
| CustomerAge | 61.82 |
| TransactionDuration | 117.94 |
| LoginAttempts | 1.13 |
| AccountBalance | 4466.27 |
| TransactionType | Debit |
| Channel | ATM |
| CustomerOccupation | Retired |

This cluster is dominated by senior or retired customers with moderate account balances. They prefer using ATMs for convenience and perform simpler financial activities. They value stability and security.

**Business Strategy:** Recommend retirement savings plans, deposit accounts, insurance products, and health protection services emphasizing safety and reliability.

### Strategy Summary

| Cluster | Segment Description | Recommended Strategy |
| :--- | :--- | :--- |
| **1** | Established professionals with stable financial activity | Premium investment & personal banking services |
| **2** | Young customers with growth potential | Financial literacy & digital engagement programs |
| **3** | Seniors with moderate stability needs | Safe, stable, and protection-focused products |
