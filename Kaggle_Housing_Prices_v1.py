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

train_file = os.path.join(os.path.dirname(__file__), 'train.csv')
test_file = os.path.join(os.path.dirname(__file__), 'test.csv')
submission_file = os.path.join(os.path.dirname(__file__), 'submission.csv')



