from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, SimpleRNN
import numpy as np


def CalcError(arr1, arr2):
     el = 0
     for i in range(5000):
         e = abs(((arr1[i] - arr2[i]) / arr1[i])) * 100
         el += e
     return el / 5000

y_fixed = 0.5
x = np.linspace(-2,1,5000)
X, Y = np.meshgrid(x, [y_fixed])
Z = 0.5 * np.sin(X + Y) * np.cos(Y)
Z_fixed = 0.5 * np.sin(x + y_fixed) * np.cos(y_fixed)
Zo = []


for i in range(5000):
 Zo.append(0.5 * np.sin(x[i] + y_fixed) * np.cos(y_fixed))


X_train = np.column_stack((X.ravel(), Y.ravel()))
y_train = Z_fixed.ravel()
Xo_train = np.column_stack((x, np.full(5000, y_fixed)))
##feed-forward
model_10_neurons = Sequential()
model_10_neurons.add(Dense(10, input_dim=2, activation='relu'))
model_10_neurons.add(Dense(1, activation='linear'))
model_10_neurons.compile(optimizer='adam', loss='mse')
history1 = model_10_neurons.fit(Xo_train, y_train, epochs=10, batch_size=10, verbose=1)
model_20_neurons = Sequential()
model_20_neurons.add(Dense(20, input_dim=2, activation='relu'))
model_20_neurons.add(Dense(1, activation='linear'))
model_20_neurons.compile(optimizer='adam', loss='mse')
history2 = model_20_neurons.fit(Xo_train, y_train, epochs=10, batch_size=10, verbose=1)
# Передбачення для обох моделей
Z_pred_10 = model_10_neurons.predict(Xo_train).ravel()
Z_pred_20 = model_20_neurons.predict(Xo_train).ravel()

##cascade-forward
########################################################################################

input_layer = Input(shape=(2,))
hidden_layer1 = Dense(20, activation='relu')(input_layer)
output = Dense(1, activation='linear')(Concatenate()([input_layer,
hidden_layer1]))
model_cascade_1 = Model(inputs=input_layer, outputs=output)
model_cascade_1.compile(optimizer='adam', loss='mse')
history3 = model_cascade_1.fit(Xo_train, y_train, epochs=10, batch_size=100, verbose=1)

################################################################################################

input_layer2 = Input(shape=(2,))
hidden_layer_1 = Dense(10, activation='relu')(input_layer)
hidden_layer_2 = Dense(10, activation='relu')(Concatenate()([input_layer,
hidden_layer1]))
output_layer = Dense(1, activation='linear')(Concatenate()([input_layer,
hidden_layer_1, hidden_layer_2]))
model_cascade_2 = Model(inputs=input_layer, outputs=output_layer)
model_cascade_2.compile(optimizer='adam', loss='mse')
history4 = model_cascade_2.fit(Xo_train, y_train, epochs=10, batch_size=10,
verbose=1)
Z_pred_1 = model_cascade_1.predict(Xo_train).ravel()
Z_pred_2 = model_cascade_2.predict(Xo_train).ravel()

####################################################################################################################


X_train_rnn = Xo_train.reshape(-1, 1, 2)

## Моделі Elman
model_elman_1 = Sequential()
model_elman_1.add(SimpleRNN(15, activation='relu', input_shape=(1, 2), return_sequences=False))
model_elman_1.add(Dense(1, activation='linear'))
model_elman_1.compile(optimizer='adam', loss='mse')
history5 = model_elman_1.fit(X_train_rnn, y_train, epochs=100, batch_size=10, verbose=1)
model_elman_3 = Sequential()
model_elman_3.add(SimpleRNN(5, activation='relu', input_shape=(1, 2), return_sequences=True))
model_elman_3.add(SimpleRNN(5, activation='relu', return_sequences=True))
model_elman_3.add(SimpleRNN(5, activation='relu', return_sequences=False))
model_elman_3.add(Dense(1, activation='linear'))
model_elman_3.compile(optimizer='adam', loss='mse')
history6 = model_elman_3.fit(X_train_rnn, y_train, epochs=20, batch_size=10, verbose=1)

# Передбачення для Elman моделей

Xo_train_rnn = Xo_train.reshape(-1, 1, 2)
Z_pred_1_elman = model_elman_1.predict(Xo_train_rnn).ravel()
Z_pred_2_elman = model_elman_3.predict(Xo_train_rnn).ravel()

####################################################################################################################

plt.plot(x,Zo, label='Функція, що модулюється')
plt.plot(x, Z_pred_10, label="10 нейронів")
plt.plot(x, Z_pred_20, label="20 нейронів")
plt.legend()
plt.title('feed-forward')
plt.show()
print("Для 10 нейронів:",CalcError(Z_pred_10, Zo))
print("Для 20 нейронів:",CalcError(Z_pred_20, Zo))
plt.plot(history1.history['loss'], label='Помилка на тренуванні 10 нейронів')
plt.plot(history2.history['loss'], label='Помилка на тренуванні 20 нейронів')
plt.title('Залежність помилки від кількості епох')
plt.xlabel('Епоха')
plt.ylabel('Помилка (MSE)')
plt.legend()
plt.show()
plt.plot(x,Zo, label='Функція, що модулюється')
plt.plot(x, Z_pred_1, label="cascade - forward backprop, 1 шар, 20 нейронів")
plt.plot(x, Z_pred_2, label="cascade - forward backprop, 2 шари по 10 нейронів")
plt.legend()
plt.show()
print("Для 1 шару, 20 нейронів:",CalcError(Z_pred_1, Zo))
print("Для 2 шари, 10 нейронів:",CalcError(Z_pred_2, Zo))
plt.plot(history3.history['loss'], label='cascade - forward backprop, 1 шар, 20 нейронів')
plt.plot(history4.history['loss'], label='cascade - forward backprop, 2 шари, 10 нейронів')
plt.title('Залежність помилки від кількості епох')
plt.xlabel('Епоха')
plt.ylabel('Помилка (MSE)')
plt.legend()
plt.show()
plt.plot(x,Zo, label='Функція, що модулюється')
plt.plot(x, Z_pred_1_elman, label="elman backprop, 1 шар, 15 нейронів")
plt.plot(x, Z_pred_2_elman, label="elman backprop, 3 шари по 5 нейронів")
plt.legend()
plt.show()
print("Для 1 шар, 15 нейронів:",CalcError(Z_pred_1_elman, Zo))
print("Для 3 шар, 5 нейронів:",CalcError(Z_pred_2_elman, Zo))
plt.plot(history5.history['loss'], label='elman backprop, 1 шар, 15 нейронів')
plt.plot(history6.history['loss'], label='elman backprop, 3 шари, 5 нейронів')
plt.title('Залежність помилки від кількості епох')
plt.xlabel('Епоха')
plt.ylabel('Помилка (MSE)')
plt.legend()
plt.show()