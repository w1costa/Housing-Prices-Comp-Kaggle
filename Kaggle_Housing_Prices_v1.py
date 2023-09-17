''' Housing Prices Competition - Machine Learning Tool to predict the sales price of each house. 
Created by Wanda Costa, on 12/06/2023

Input - The train set has 1460 observations and 81 features.
Output - For each Id in the test set, you must predict the value of the SalePrice variable.

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and 
the logarithm of the observed sales price. 

'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

train_file = os.path.join(os.path.dirname(__file__), 'train.csv')
test_file = os.path.join(os.path.dirname(__file__), 'test.csv')
submission_file = os.path.join(os.path.dirname(__file__), 'submission.csv')

# Read the data
X_full = pd.read_csv(train_file, index_col='Id')
X_test_full = pd.read_csv(test_file, index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Fill in the line below: get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

print("MAE (Drop columns with missing values):")
print(round(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid), 2))

#Add Imputer

# Fill in the lines below: imputation
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(round(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid), 2))

# Preprocessed training and validation features
final_X_train = reduced_X_train
final_X_valid = reduced_X_valid

# Imputation
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# Define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(round(mean_absolute_error(y_valid, preds_valid), 2))

#X_test.info()
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
# Fill in the line below: preprocess test data
final_X_test = pd.DataFrame(imputer.fit_transform(reduced_X_test))
#final_X_test.info()

# Fill in the lines below: imputation removed column names; put them back
final_X_test.columns = reduced_X_test.columns

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
