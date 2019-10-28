from sklearn.decomposition import PCA
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import FastICA
from sklearn.metrics import classification_report, confusion_matrix

ica = FastICA(n_components=5)

base = pd.read_csv('cardio_train.csv')
caracteristicas = ['age', 'gender', 'height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
age = []
for i in base.age.values:
    i = i//365
    age.append(i)

base['age'] = age
x = base[caracteristicas]
y = base.cardio

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

x_treino = ica.fit_transform(x_treino)
x_teste = ica.transform(x_teste)

classificadorArvore = DecisionTreeClassifier(criterion="entropy", max_depth=3,splitter='best')

classificadorArvore = classificadorArvore.fit(x_treino,y_treino)


y_pred = classificadorArvore.predict(x_teste)


print("Precisao: ",metrics.accuracy_score(y_teste, y_pred))


print(classification_report(y_teste, y_pred))
print('\n')
print(confusion_matrix(y_teste, y_pred))