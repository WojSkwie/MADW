#import tensorflow.contrib.keras as keras
import keras as keras
from keras.layers import Dense
import matplotlib.pyplot as plt
from data import ClassificationTwoSpiralsData, input_noise,\
    label_noise, prepare_grid_points_for_contour_plot
import time

DRAW = True

plt.ion()

# data generation parameters
NPOINTS = 1000
MIN_X = -12.
MAX_X = 12.
GRID_SIZE = 20

# training hyperparameters
LR = 0.1

# generate data
data_gen = ClassificationTwoSpiralsData(min_x=MIN_X, max_x=MAX_X)
data_x, data_targets = data_gen.generate_data(n_points=NPOINTS)

data_x = input_noise(data_x)
data_targets = label_noise(data_targets)


model = keras.models.Sequential()
#model.add(Dense(64, activation='tanh'))
model.add(Dense(units=20, input_dim=2, activation='relu'))
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batchsize = 128


if DRAW:
    ax = plt.subplot(111)
    X1, X2, all_x_pairs = prepare_grid_points_for_contour_plot(MIN_X, MAX_X, GRID_SIZE)

class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, show_modulo_epoch=10):
        super().__init__()
        self.modulo = show_modulo_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.modulo == 0:
            decision_values = model.predict(x = all_x_pairs, batch_size=batchsize)
            Z = decision_values.reshape((GRID_SIZE, GRID_SIZE))
            data_gen.plot_data(ax, data_x, data_targets,
                               x_grid=X1, y_grid=X2, z_model=Z, add_bar=False,
                               title='Epoch {}'.format(epoch))
            plt.pause(0.01)


clbks = [keras.callbacks.TensorBoard(log_dir='./tb_log/'+time.strftime('%c'),
            write_graph=True)]

if DRAW:
    clbks.append(VisualizationCallback(show_modulo_epoch=5))

model.fit(data_x, data_targets, epochs=500, batch_size=batchsize, callbacks=clbks)
