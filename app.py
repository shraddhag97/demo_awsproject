import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")

# Feature engineering
df.drop("Id", axis=1, inplace = True)
df["Species"].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}, inplace = True)

#Train Model
x = df.drop("SepalLengthCm", axis=1) # independent vars
y = df["SepalLengthCm"] # dependent var

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle = True, random_state= 1)
linear_reg_model = LinearRegression()
linear_reg_model.fit(x_train, y_train)

y_pred_train = linear_reg_model.predict(x_train)
y_pred_test = linear_reg_model.predict(x_test)


def predict_sepal(SepalWidthCm, PetalLengthCm, PetalWidthCm, Species):
    if Species == 'Iris-setosa':
        Species = 0
    elif Species == 'Iris-versicolor':
        Species = 1
    elif Species == 'Iris-virginica':
        Species = 2
    df_test = pd.DataFrame({"SepalWidthCm":[SepalWidthCm], "PetalLengthCm" : [PetalLengthCm], 
                           "PetalWidthCm" : [PetalWidthCm], "Species" : [Species]})
    test_pred = linear_reg_model.predict(df_test)
    sepal_length = np.around(test_pred[0],2)
    
    print(f"Predicted SepalLengthCm value is :{sepal_length} cm")
    return sepal_length

