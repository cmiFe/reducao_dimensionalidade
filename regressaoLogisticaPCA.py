#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:34:37 2019

@author: guilherme
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
base = pd.read_csv('cardio_train.csv')
caracteristicas = ['age', 'gender', 'height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
age = []
x = base[caracteristicas]
y = base.cardio
for i in base.age.values:
    i = i//365
    age.append(i)

base['age'] = age

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

x_treino = pca.fit_transform(x_treino)
x_teste = pca.transform(x_teste)
classificadorLogistico = LogisticRegression()

classificadorLogistico.fit(x_treino,y_treino)



y_pred=classificadorLogistico.predict(x_teste)


print("Precisao: ",metrics.accuracy_score(y_teste, y_pred))