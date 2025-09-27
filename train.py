import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("quikr_car.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df['year'].unique()
for i in df.columns:
    df[i].unique()
    print(df[i],"\n",df[i].unique())
df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
df['year']= pd.to_numeric(df['year'],errors='coerce')
df['year'] = df['year'].fillna(df['year'].median())
df['year'].shape
df = df[df['Price']!="Ask For Price"]
df['Price'] = df['Price'].str.replace(",","").astype(int)
df['kms_driven']=df['kms_driven'].str.split(" ").str.get(0).str.replace(",","")
df = df[df['kms_driven'].str.isnumeric().fillna(False)]
df['kms_driven']=df['kms_driven'].astype(int)
df = df[~df['fuel_type'].isna()]
df['name'].str.split(" ").str.slice(0,3).str.join(" ")
df.reset_index(drop=True)
df.isnull().sum()
df.describe()
df = df[df['Price'] < 6000000].reset_index(drop=True)
df.describe()
df.shape
df.to_csv("cleaned_car.csv")
df.head()
X = df.drop(columns=['Price'])
Y = df['Price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(X.shape,X_train.shape,X_test.shape)
df.head()
numeric_features = ['year','kms_driven']
categorical_features = ['company','fuel_type','name']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ]
)
#make pipeline 

pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',LinearRegression())
])

pipeline.fit(X_train,Y_train)
import joblib

Y_pred = pipeline.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2

import os
import pickle


# Save the pipeline using pickle
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)