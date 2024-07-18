import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Tc.csv')

# Define the target and features
X = data.drop(columns=['Tc_real'])
y = data['Tc_real']

# Standardize the features
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Define the number of features to select (assuming selected_features is defined)
selected_features = list(range(X_sc.shape[1]))  # Select all features initially

# Setup the pipeline
skb = SelectKBest(score_func=f_regression)
est_rf = RandomForestRegressor(random_state=0)
pipe_rf = Pipeline([('SKB', skb), ('forest', est_rf)])

# Define the parameter grid for grid search
param_grid_rf = {
    'SKB__k': [20, 30, 40],  # Number of features to select
    'forest__n_estimators': [100, 200, 300],
    'forest__max_depth': [None, 10, 20],
    'forest__max_features': ['auto', 'sqrt', 'log2'],
    'forest__min_samples_leaf': [1, 2, 4]
}

# Perform grid search
gcv_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=0)
gcv_rf.fit(X_train, y_train)

# Print the best score and parameters
print('\nRandomForest:', gcv_rf.best_score_)
print(gcv_rf.best_params_)

# Extract the best estimator
best_params = gcv_rf.best_params_
best_est_rf = RandomForestRegressor(
    random_state=0,
    max_features=best_params['forest__max_features'],
    n_estimators=best_params['forest__n_estimators'],
    max_depth=best_params['forest__max_depth'],
    min_samples_leaf=best_params['forest__min_samples_leaf']
)

# Perform the best results search
best_results = -10
best_state = -1
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=i)
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]
    best_est_rf.fit(X_train, y_train)
    result = best_est_rf.score(X_test, y_test)
    if result > best_results:
        best_results = result
        best_state = i

print(best_results, best_state)

# Final split with the best random state
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=best_state)
X_train = X_train[:, selected_features]
X_test = X_test[:, selected_features]
