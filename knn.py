from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


base = pd.read_csv('cardio_train.csv')
caracteristicas = ['age', 'gender', 'height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
age = []

sc = StandardScaler()

x = base[caracteristicas]
y = base.cardio

#sc.fit_transform(x)

for i in base.age.values:
    i = i//365
    age.append(i)

base['age'] = age
fig, ax = plt.subplots()

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
#x_treino = sc.fit_transform(x_treino)
#x_teste = sc.transform(x_teste)

#ax.scatter(x,x_treino,color='red')
#ax.scatter(x_treino,color='red')
# set a title and labels
#plt.show()
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(x_treino,y_treino)
y_pred = modelo.predict(x_teste)
print("Accuracy:",metrics.accuracy_score(y_teste, y_pred))
# Train the model using the training sets
print(classification_report(y_teste, y_pred))
print('\n')
print(confusion_matrix(y_teste, y_pred))