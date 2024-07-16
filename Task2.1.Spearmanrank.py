import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import numpy as np
from scipy.cluster.hierarchy import ward
from scipy.spatial.distance import squareform

# Load the dataset (adjust the path as needed)
data = pd.read_csv('/Users/sudeerde/Desktop/Machine Learning /Tc.csv')

# Convert categorical data to numeric using one-hot encoding
data = pd.get_dummies(data)

# Assuming the last column is the target property and we exclude it from features
features = list(data.columns)[:-1]
X = data[features]

# Remove features with zero variance
X = X.loc[:, (X != X.iloc[0]).any()]

# Normalize the feature data
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Calculate the Spearman rank-order correlation coefficient
corr, _ = spearmanr(X_sc, axis=0)

# Handle potential NaN values in the correlation matrix
corr = np.nan_to_num(corr)

# Print the shape of the correlation matrix
print('Correlation matrix shape:', corr.shape)

# Convert the correlation matrix to a condensed distance matrix
# (1 - correlation) because Ward linkage requires a distance metric
dist_matrix = 1 - corr

# Ensure the distance matrix is symmetric
dist_matrix = (dist_matrix + dist_matrix.T) / 2

# Set the diagonal to zero to indicate no distance with itself
np.fill_diagonal(dist_matrix, 0)

# Convert to condensed form (needed for hierarchical clustering)
condensed_dist_matrix = squareform(dist_matrix)

# Perform hierarchical clustering using Ward's method
corr_linkage = ward(condensed_dist_matrix)

# Print the hierarchical clustering results
print(corr_linkage)



###on Terminal , Each row of the linkage matrix represents a merge in the hierarchical clustering.
#The first two columns represent the indices of the clusters being merged.
#The third column represents the distance between the clusters being merged.
#The fourth column represents the number of original observations in the newly formed cluster.
