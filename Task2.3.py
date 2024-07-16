import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/sudeerde/Desktop/Machine Learning /Tc.csv')  # Provide the correct path to your dataset
print('Dataset shape:', data.shape)
print(data.describe())

# Convert categorical data to numeric using one-hot encoding
data = pd.get_dummies(data)

# Assuming the last column is the target property
features = list(data.columns)[:-1]  # All columns except the last one
target = data.columns[-1]           # The last column

# Shuffle the data
data = shuffle(data, random_state=0)

# Split into features (X) and target (y)
X = data[features]
y = data[target]

print('Number of features:', len(features))

# Initialize a Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=0)

# Perform cross-validation
scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')

# Output the mean score
print('Mean CV score:', scores.mean())

# Variance Threshold for feature selection
sel = VarianceThreshold(threshold=(0.95 * (1 - 0.95)))

# Apply the variance threshold
X_sel = sel.fit_transform(X)

# Output results
feature_index = sel.get_support()
print('Number of features after removing redundant ones:', sum(feature_index))

# List of final features
final_features = [features[i] for i in range(len(features)) if feature_index[i]]
print('Final selected features:', final_features)

# Normalize the data
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_sel)

# Print the shape of the normalized data
print('X_sc shape:', X_sc.shape)

# Perform cross-validation with the normalized data
scores_normalized = cross_val_score(reg, X_sc, y, cv=5, scoring='neg_mean_squared_error')

# Output the mean score for normalized data
print('Mean CV score with normalized data:', scores_normalized.mean())

# Correlation Analysis using Spearman rank-order correlation
corr = spearmanr(X_sc).correlation
print('Correlation matrix shape:', corr.shape)

# Hierarchical clustering and dendrogram
corr_linkage = hierarchy.ward(corr)  # Perform hierarchical clustering using the ward method

# Create and plot the dendrogram
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
dendro = hierarchy.dendrogram(corr_linkage, labels=final_features, ax=ax)
plt.title("Dendrogram")
plt.xlabel("Features")
plt.ylabel("Distance")
plt.show()

# Print leaves and labels
print(dendro['leaves'])
print(dendro['ivl'])
Task 2.3: Visualize correlations with a heatmap
ordered_corr = corr[dendro['leaves'], :][:, dendro['leaves']]

plt.figure(figsize=(10, 8))
sns.heatmap(ordered_corr, xticklabels=np.array(final_features)[dendro['leaves']], yticklabels=np.array(final_features)[dendro['leaves']], cmap='coolwarm', annot=False)
plt.title("Heatmap of Correlations Ordered by Hierarchical Clustering")
plt.colorbar()
plt.show()

