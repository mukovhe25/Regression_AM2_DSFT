"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')

train = train[(train['Commodities'] == 'APPLE GOLDEN DELICIOUS')]

y_train = train['avg_price_per_kg']
X_train = train[['Weight_Kg', 'Low_Price', 'High_Price', 'Sales_Total', 'Total_Qty_Sold', 'Total_Kg_Sold', 'Stock_On_Hand']]

# Fit model
GBR = GradientBoostingRegressor(n_estimators = 280, min_samples_leaf = 4, max_depth = 7, learning_rate = 0.19183673469387755, random_state = 42)
print ("Training Model...")
GBR.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/apples_simple_GBR.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(GBR, open(save_path,'wb'))
