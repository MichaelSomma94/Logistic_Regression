# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:59:06 2022

@author: Michael
"""
import numpy as np
import pandas as pd
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
pd.set_option('display.max_columns', None)

#

data = pd.read_csv("Heart_Disease_Prediction.csv")
data = data.dropna()
data["Heart Disease"] = data["Heart Disease"].map({"Presence": 1, "Absence": 0})



# figure = px.scatter(data_frame = data, x="Age",
#                     y="Heart Disease", trendline="ols")
# figure.show()

x = np.array(data.iloc[:, 0:13])
y = np.array(data['Heart Disease'])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)


LogisticRegr = LogisticRegression(random_state=42, max_iter=100)

logisticRegr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
logisticRegr = logisticRegr.fit(xtrain, ytrain.ravel())
classifier = logisticRegr.named_steps['logisticregression']



for i in range(len(data.columns.to_numpy())-1):
    print(data.columns.to_numpy()[i], '' ,  classifier.coef_[0,i] )

ypred=logisticRegr.predict(xtest)
print(ytest.ravel())
print(ypred)
print(logisticRegr.score(xtest, ytest))
cm = confusion_matrix(ytest, ypred)
print(cm)
scores = cross_val_score(logisticRegr, x, y.ravel(), cv=5)
print(np.mean(scores), "+/-", np.std(scores))
