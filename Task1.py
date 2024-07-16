import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.feature_selection import VarianceThreshold

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
# Get features that return at least 95% of the Bernoulli random variables
sel = VarianceThreshold(threshold=(0.95 * (1 - 0.95)))

# Apply the variance threshold
X_sel = sel.fit_transform(X)

# Output results
feature_index = sel.get_support()
print('Number of features after removing redundant ones:', sum(feature_index))

# List of final features
final_features = []

for i in range(len(features)):
    if feature_index[i]:
        final_features.append(features[i])

print('Final selected features:', final_features)

# Use the selected features for the final model training
X_selected = data[final_features]

# Initialize a Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=0)

# Perform cross-validation
scores = cross_val_score(reg, X_selected, y, cv=5, scoring='neg_mean_squared_error')

# Output the mean score
print('Mean CV score with selected features:', scores.mean())

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model
grid_search.fit(X_selected, y)

# Best parameters and score
print('Best parameters found: ', grid_search.best_params_)
print('Best cross-validation score: ', grid_search.best_score_)
# Train the final model with the best parameters
best_reg = grid_search.best_estimator_

# Fit the final model on the entire dataset
best_reg.fit(X_selected, y)

# Evaluate the final model (if you have a test set, you can use it here)
# For demonstration, using cross-validation again
final_scores = cross_val_score(best_reg, X_selected, y, cv=5, scoring='neg_mean_squared_error')

# Output the final mean score
print('Final mean CV score with tuned parameters:', final_scores.mean())
feature_importances = pd.DataFrame(best_reg.feature_importances_, index=X_selected.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
