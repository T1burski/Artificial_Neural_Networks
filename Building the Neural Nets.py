import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
np.random.seed(42)
tf.random.set_seed(42)

%matplotlib inline

#Catching all the data we will need as inputs and outputs. In order to work with a dataset that has more chances 
#to produce overfitting, we use the data produced by the K-Means previously.

df_validacao = pd.read_csv('Dados_Python_5_K_Means_0_Teste.csv')
df_treino = pd.read_csv('Dados_Python_5_K_Means_2_Treino.csv')
df_teste = pd.read_csv('Dados_Python_5_K_Means_1_Validacao.csv')

x_input_teste = df_teste.drop(['T_agua_fria'], axis=1)
y_output_teste = df_teste[['T_agua_fria']]

x_input_treino = df_treino.drop(['T_agua_fria'], axis=1)
y_output_treino = df_treino[['T_agua_fria']]

x_input_validacao = df_validacao.drop(['T_agua_fria'], axis=1)
y_output_validacao = df_validacao[['T_agua_fria']]

#Using z-score normalization

def scale_datasets(x_input_treino, x_input_teste):
    standard_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        standard_scaler.fit_transform(x_input_treino),
        columns= x_input_treino.columns
    )
    X_test_scaled = pd.DataFrame(
        standard_scaler.transform(x_input_teste),
        columns = x_input_teste.columns
    )
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scale_datasets(x_input_treino, x_input_teste)

def scale_datasets_v(x_input_validacao):
    standard_scaler = StandardScaler()
    X_validation_scaled = pd.DataFrame(
        standard_scaler.fit_transform(x_input_validacao),
        columns= x_input_validacao.columns
    )
    return X_validation_scaled
X_validation_scaled = scale_datasets_v(x_input_validacao)

#Converting the datasets to numpy arrays in order to use Keras afterwards.

X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)
X_validation_scaled = np.array(X_validation_scaled)
y_train = np.array(y_output_treino)
y_test = np.array(y_output_teste)
y_validation = np.array(y_output_validacao)


#Initializing the training of the neural network. Below, an example using two hidden layers with 5 and 4 hidden nodes

model = Sequential()
model.add(Dense(5, input_dim=6, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1))


#Below, the construction of the neural network when regularization is used

#model = Sequential()
#model.add(Dense(5, input_dim=6, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01597)))
#model.add(Dropout(0.9))
#model.add(Dense(4, activation="relu"))#, kernel_regularizer=keras.regularizers.l2(0.3)))
#model.add(Dropout(0.9))
#model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(x_input_treino, y_train, epochs=2500, batch_size=64, verbose=1, validation_data=(x_input_validacao, y_validation), shuffle = False)

#Applying the trained neural net to the test dataset and printing MAE and RMSE

y_predict = model.predict(x_input_teste)
print("T_agua_fria MAE:%.4f" % mean_absolute_error(y_test[:,0], y_predict[:,0]))
print("T_agua_fria RMSE:%.4f" % mean_squared_error(y_test[:,0], y_predict[:,0], squared=False))


#Code used to output the learning curve of the neural net using the loss function: MAE

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

#Code used to output the learning curve of the neural net using the RMSE

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model root_mean_squared_error')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()