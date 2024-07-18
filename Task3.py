import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hierarchy
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load data from CSV file
data = pd.read_csv('C:/Users/mahta/Desktop/Machine Learning Project/Machine-Learning-Project/Tc.csv')

# Define the target column
target_column = 'Tc_real'

# Ensure all features are numeric
X = data.drop(target_column, axis=1)

# Convert all columns to numeric, forcing errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Drop any columns that contain NaN values
X = X.dropna(axis=1, how='any')

# Extract the target column
y = data[target_column]

# Perform feature selection using hierarchical clustering
# Compute the correlation matrix
corr = np.corrcoef(X.values, rowvar=False)

# Replace any NaN or infinite values in the correlation matrix with zero
corr = np.nan_to_num(corr)

# Verify that all values are finite
if not np.all(np.isfinite(corr)):
    print("Error: The correlation matrix contains non-finite values.")
    # Replace remaining infinite values with large finite numbers
    corr[~np.isfinite(corr)] = 0

# Compute the linkage matrix
corr_linkage = hierarchy.linkage(corr, method='average')

# Form flat clusters
cluster_ids = hierarchy.fcluster(corr_linkage, t=2, criterion='distance')

# Map clusters to feature indices
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

# Select the first feature from each cluster
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

# Print the selected features and their count
print("Selected features:", selected_features)
print('Number of features after correlation reduction:', len(selected_features))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, selected_features], y, test_size=0.1, random_state=0)

# Set up SelectKBest and RandomForestRegressor
skb = SelectKBest(score_func=f_regression)
est_rf = RandomForestRegressor(random_state=0)
pipe_rf = Pipeline([('SKB', skb), ('forest', est_rf)])

# Define the hyperparameter grid
param_grid_rf = {
    'SKB__k': [5, 10, 15, 20],  # Number of features to select
    'forest__n_estimators': [100, 200, 300],  # Number of trees
    'forest__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'forest__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
    'forest__min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Create the GridSearchCV object
gcv_rf = GridSearchCV(estimator=pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

# Fit the model
gcv_rf.fit(X_train, y_train)

# Print the best score and parameters
print('\nRandomForest:', gcv_rf.best_score_)
print(gcv_rf.best_params_)

# Evaluate the model on the test set
score = gcv_rf.score(X_test, y_test)
print(f'Test set score: {score}')


