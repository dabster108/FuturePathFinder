# Pre-Modeling Feature Analysis Report

This document outlines the comprehensive pre-modeling analysis performed on the `cleaned_education_data.csv` dataset. The objective of this analysis is to understand feature characteristics, identify influential variables, and prepare the data for robust model development.

## 1. Feature Importance Analysis

To identify the most predictive features for our regression (`Starting_Salary`) and classification (`Career_Domain`) tasks, we employed two primary techniques.

### Correlation Matrix

A correlation matrix was generated to visualize the linear relationships between numerical features and the `Starting_Salary`. This helps in identifying potential multicollinearity and features that have a strong linear correlation with the target.

![Correlation Matrix](feature_visualizations/correlation_matrix.png)

### ExtraTrees-Based Feature Importance

To capture both linear and non-linear relationships, we utilized ExtraTrees (Extremely Randomized Trees) models. This powerful ensemble method provides importance scores for each feature.

**For Regression (Predicting `Starting_Salary`):**
The plot below shows the feature importance scores for predicting the starting salary.

![Feature Importance for Regression](feature_visualizations/feature_importance_regression.png)

**For Classification (Predicting `Career_Domain`):**
This plot ranks features based on their ability to predict the career domain.

![Feature Importance for Classification](feature_visualizations/feature_importance_classification.png)

## 2. Target Variable Distribution

Understanding the distribution of the target variables is crucial for selecting the right modeling approach and evaluation metrics.

### Regression Target: `Starting_Salary`

We plotted a histogram to analyze the distribution of `Starting_Salary`. This helps in assessing normality and identifying any skewness that might need transformation.

![Starting Salary Distribution](feature_visualizations/regression_target_distribution.png)

### Classification Target: `Career_Domain`

A bar plot was used to examine the class balance for the `Career_Domain`. This is critical for identifying class imbalance, which can bias the model if not handled properly.

![Class Balance for Career Domain](feature_visualizations/classification_target_balance.png)

## 3. Feature Relationship Visualization

Exploring the interactions between features can reveal complex patterns that are not apparent from univariate analysis.

### Pairwise Relationships

A pairplot was created to visualize the pairwise relationships between key numerical features, with data points colored by the `Career_Domain`. This provides insights into how features interact within different classes.

![Key Features Pairplot](feature_visualizations/key_features_pairplot.png)

### Interaction Heatmap

To understand how two categorical features jointly affect a numerical target, we generated a heatmap showing the mean `Starting_Salary` based on `Field_of_Study` and `Interest`.

![Salary Interaction Heatmap](feature_visualizations/salary_interaction_heatmap.png)

## 4. Dimensionality Reduction Visualization

To visualize the structure of the high-dimensional numerical feature space, we used PCA and t-SNE to project the data into two dimensions.

### Principal Component Analysis (PCA)

PCA provides a linear projection of the data, highlighting the variance and separation between classes.

![PCA Visualization](feature_visualizations/pca_visualization.png)

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear technique that is particularly effective at revealing underlying clusters or manifolds in the data.

![t-SNE Visualization](feature_visualizations/tsne_visualization.png)

## 5. Business Rule Validation

To enhance data quality, we applied a domain-specific rule to filter out illogical combinations of `Field_of_Study` and `Interest`. This step ensures that the data used for modeling reflects real-world constraints. The analysis showed that a number of rows were removed, improving the logical consistency of the dataset.

## 6. Train-Test Split Preparation

The final step in our pre-modeling analysis was to prepare the data for model training. We performed a **stratified train-test split**. Stratification ensures that the distribution of the `Career_Domain` target variable is preserved in both the training and testing sets. This is crucial for building a generalizable and reliable classification model, especially given the observed class imbalances.
