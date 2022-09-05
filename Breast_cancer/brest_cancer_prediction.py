# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:29:50 2022

@author: Michael
"""
import pandas as pd
import numpy as np
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

data = pd.read_csv("BRCA.csv")
data = data.dropna()


data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"] = data["PR status"].map({"Positive": 1})
data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
data["Gender"] = data["Gender"].map({"MALE": 0, "FEMALE": 1})
data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4})
data["Patient_Status"] = data["Patient_Status"].map({'Alive': 0, "Dead": 1})
x = np.array(data[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                   'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                   'HER2 status', 'Surgery_type']])
y = np.array(data[['Patient_Status']])
xtrain, xtest, ytrain, ytest = train_test_split(StandardScaler().fit_transform(x), y, test_size=0.10, random_state=42)


logisticRegr = LogisticRegression(random_state=42, max_iter=100)


#ytrain = np.reshape(ytrain, (1,np.product(ytrain.shape)))[0]
#min_max_scaler = preprocessing.MinMaxScaler()
xtrain = preprocessing.StandardScaler().fit(xtrain)
logisticRegr = LogisticRegression(random_state=42)
logisticRegr.fit(xtrain, ytrain.ravel())

ypred=logisticRegr.predict(xtest)
print(ytest.ravel())
print(ypred)
print(logisticRegr.score(xtest, ytest))
cm = confusion_matrix(ytest, ypred)
print(cm)

scores = cross_val_score(logisticRegr, x, y.ravel(), cv=5)

print(np.mean(scores), "+/-", np.std(scores))
# coef = logisticRegr.coef_
# print (coef)
