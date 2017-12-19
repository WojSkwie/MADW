import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import ClassificationLinearData, input_noise, label_noise

plt.ion()

# data generation parameters
NPOINTS = 1000
MIN_X = -5.
MAX_X = 5.
GRID_SIZE = 20

# training hyperparameters
LR = 0.1

# run parameters
DRAW = False


# generate data
data_gen = ClassificationLinearData(min_x=MIN_X, max_x=MAX_X)
data_x, data_targets = data_gen.generate_data(n_points=NPOINTS)

data_x = input_noise(data_x)
data_targets = label_noise(data_targets)


if DRAW:
    ax = plt.subplot(111)
    data_gen.plot_data(ax, data_x, data_targets)
    plt.show(block=True)


# creating default graph
# TODO: define graph for sigmoid neuron with two inputs


if DRAW:
    ax = plt.subplot(111)
    values = np.linspace(MIN_X, MAX_X, GRID_SIZE)
    X1, X2 = np.meshgrid(values, values)
    X1vec = X1.reshape((X1.size, 1))
    X2vec = X2.reshape((X2.size, 1))
    X = np.hstack((X1vec, X2vec))

# # # Launch the graph.
with tf.Session() as sess:
    # TODO: initialize variables

    for step in range(201):
        if step % 1 == 0:
            # TODO : get values of W and loss tensors
            print('step {}: w1 = {:.3f}, w2 = {:.3f}'.format(step, wvalue[0, 0], wvalue[1, 0]))
            print('loss: {}'.format(loss_value))
            if DRAW:
                decision_values = sess.run(y, feed_dict={x: X})
                Z = decision_values.reshape((GRID_SIZE, GRID_SIZE))
                add_bar = True if step == 0 else False
                data_gen.plot_data(ax, data_x, data_targets,
                                   x_grid=X1, y_grid=X2, z_model=Z, add_bar=add_bar)
                plt.pause(0.2)
        # TODO: run train_op

    sess.close()
