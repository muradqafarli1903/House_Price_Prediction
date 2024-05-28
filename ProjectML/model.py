import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle


file_path = 'House_Rent_Dataset.csv'
house_data = pd.read_csv(file_path)

house_data = house_data[['BHK', 'Size', 'Bathroom', 'City', 'Furnishing Status', 'Tenant Preferred', 'Rent','Area Type','Point of Contact']]

house_data = house_data.dropna()

X = house_data.drop(['Rent'], axis=1)
y = house_data['Rent']

numeric_features = ['BHK', 'Size', 'Bathroom']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['City', 'Furnishing Status', 'Tenant Preferred','Area Type', 'Point of Contact']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(learning_rate=0.01, max_depth =2, n_estimators= 500))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

with open('house_rent_model.pkl', 'wb') as f:
    pickle.dump(model, f)

