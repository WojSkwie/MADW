import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import ClassificationTwoSpiralsData, input_noise, label_noise

plt.ion()

# data generation parameters
NPOINTS = 1000
MIN_X = -12.
MAX_X = 12.
GRID_SIZE = 20

# training hyperparameters
LR = 0.1

# run parameters
DRAW = False


# generate data
data_gen = ClassificationTwoSpiralsData(min_x=MIN_X, max_x=MAX_X)
data_x, data_targets = data_gen.generate_data(n_points=NPOINTS)

data_x = input_noise(data_x)
data_targets = label_noise(data_targets)


if DRAW:
    ax = plt.subplot(111)
    data_gen.plot_data(ax, data_x, data_targets)
    plt.show(block=True)


# creating default graph
n_outputs = [20, 40, 40, 1]
f_act = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.identity]
names = ['hidden_layer', 'hidden_layer', 'hidden_layer', 'last_layer']


def dense_layer(input_tensor, n_inputs, n_outputs, name, f_act=tf.nn.relu):
    with tf.variable_scope(name):
        w = tf.get_variable("weights", dtype=tf.float32, shape=[n_inputs, n_outputs],
                            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable("bias", dtype=tf.float32, shape=[1, n_outputs])
        h = f_act(tf.matmul(input_tensor, w) + b)
    return h


x = tf.placeholder(tf.float32, (None, 2), name='x_placeholder')
t = tf.placeholder(tf.float32, (None, 1), name='t_placeholder')

input_t = x
n_inputs = 2

for (n, (n_outs, f_a, name)) in enumerate(zip(n_outputs, f_act, names)):
    input_t = dense_layer(input_tensor=input_t, n_inputs=n_inputs, n_outputs=n_outs, name=name+str(n), f_act=f_a)
    n_inputs = n_outs

logit = input_t
y = tf.nn.sigmoid(logit, 'y')

with tf.variable_scope('CE_loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=logit), name='cross_entropy_loss')

with tf.variable_scope('accuracy_evaluation'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y), t), tf.float32))

# train = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

init = tf.global_variables_initializer()


if DRAW:
    ax = plt.subplot(111)
    values = np.linspace(MIN_X, MAX_X, GRID_SIZE)
    X1, X2 = np.meshgrid(values, values)
    X1vec = X1.reshape((X1.size, 1))
    X2vec = X2.reshape((X2.size, 1))
    X = np.hstack((X1vec, X2vec))

# # # Launch the graph.
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('/tmp/tb_log', graph=sess.graph)
    writer.close()
    for step in range(201):
        if step % 1 == 0:
            [loss_value, accuracy_value] = sess.run([loss, accuracy], feed_dict={x: data_x, t: data_targets})
            print('Step = {: 4d}, CE loss: {:.6f}, accuracy = {:.4f}'.format(step, loss_value, accuracy_value))
            if DRAW:
                decision_values = sess.run(y, feed_dict={x: X})
                Z = decision_values.reshape((GRID_SIZE, GRID_SIZE))
                add_bar = True if step == 0 else False
                data_gen.plot_data(ax, data_x, data_targets,
                                   x_grid=X1, y_grid=X2, z_model=Z, add_bar=add_bar)
                plt.pause(0.2)
        sess.run(train, feed_dict={x: data_x, t: data_targets})

    sess.close()
