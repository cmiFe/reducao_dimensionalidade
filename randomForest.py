#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:34:37 2019

@author: guilherme
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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


classificadorFloresta = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=3)

classificadorFloresta.fit(x_treino,y_treino)



y_pred=classificadorFloresta.predict(x_teste)


print("Precisao: ",metrics.accuracy_score(y_teste, y_pred))