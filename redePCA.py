from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


pca = PCA(n_components=5)
base = pd.read_csv('cardio_train.csv')
caracteristicas = ['age', 'gender', 'height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
age =[]
for i in base.age.values:
    i = i//365
    age.append(i)
base['age'] = age

x = base[caracteristicas].values
y = base.cardio.values
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

x_treino = pca.fit_transform(x_treino)
x_teste = pca.transform(x_teste)

model = Sequential()
model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_treino, y_treino, epochs=100, batch_size=32,verbose=1)

loss_and_metrics = model.evaluate(x_teste, y_teste, batch_size=128)
print(loss_and_metrics)

