''' Housing Prices Competition - Machine Learning Tool to predict the sales price of each house. 
Created by Wanda Costa, on 12/06/2023

Input - The train set has 1460 observations and 81 features.
Output - For each Id in the test set, you must predict the value of the SalePrice variable.

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and 
the logarithm of the observed sales price. 

'''

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

train_file = os.path.join(os.path.dirname(__file__), 'train.csv')
test_file = os.path.join(os.path.dirname(__file__), 'test.csv')
submission_file = os.path.join(os.path.dirname(__file__), 'submission.csv')

# Read the data
X_full = pd.read_csv(train_file, index_col='Id')
X_test_full = pd.read_csv(test_file, index_col='Id')

# Remove rows with missing target, define target, and separate target from features
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Only keep numerical features ('object' is the type used to refer to strings)
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Fill in the line below: get names of columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

# Fill in the lines below: drop columns in training and validation data
reduced_X = X.drop(cols_with_missing, axis=1)

# Imputation
final_imputer = SimpleImputer(strategy='median')
final_X = pd.DataFrame(final_imputer.fit_transform(reduced_X))

# Imputation removed column names; put them back
final_X.columns = reduced_X.columns

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X, y)

# Define cross-validation
scores = -1 * cross_val_score(model, final_X, y, cv=5, scoring='neg_mean_absolute_error')

# Print mean score and standard deviation
print("MAE scores:\n", scores)
print("Average MAE score (across experiments):")
print(round(scores.mean(), 2))
print("Standard deviation of MAE scores:")
print(round(scores.std(), 2))

# Preprocess test data
reduced_X_test = X_test.drop(cols_with_missing, axis=1)

# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(final_imputer.fit_transform(reduced_X_test))

# Make sure test data has the same order of columns as training data
final_X_test.columns = reduced_X_test.columns

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_test})
output.to_csv(submission_file, index=False)
