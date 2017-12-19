import numpy as np
import math
import matplotlib.pyplot as plt


class RegressionLinearData:
    def __init__(self, a_coeff, b_coeff):
        self.a = a_coeff
        self.b = b_coeff

    def generate_data(self, n_examples=100):
        n_features = 1
        x_data = np.random.rand(n_examples, n_features).astype(np.float32)
        y_data = x_data * self.a + self.b
        return x_data, y_data

    def plot_data(self, ax, x, y, plot_ideal_line=False, model_a=None, model_b=None):
        plt.sca(ax)
        plt.cla()
        ax.scatter(x, y, label='Data')
        xlims = np.asarray([np.min(x)-0.1, np.max(x)+0.1], dtype=np.float32)
        ylims = np.asarray([np.min(y)-0.01, np.max(y)+0.01], dtype=np.float32)
        if plot_ideal_line:
            pred_y = xlims * self.a + self.b
            plt.plot(xlims, pred_y, 'b', label='Ideal')
        if model_a is not None and model_b is not None:
            pred_y_model = xlims * model_a + model_b
            plt.plot(xlims, pred_y_model, 'r', label='Modelled')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.legend(loc='lower right')
        plt.draw()


class ClassificationLinearData:
    def __init__(self, a_coeff=0.2, b_coeff=-0.7, c_coeff=0, min_x=-5, max_x=5):
        self.a = a_coeff
        self.b = b_coeff
        self.c = c_coeff
        self.min_x = min_x
        self.max_x = max_x

    def generate_data(self, n_points=1000):
        """Simple two-class classes linearly separable by line a*x_1 + b*x_2 + c = 0."""

        datax = (self.max_x-self.min_x) * np.random.rand(n_points * 2).reshape((n_points, 2)) + self.min_x
        datax.astype('float32')
        labels = datax[:, 0] * self.a + datax[:, 1] * self.b + self.c > 0
        labels = labels.astype('float32').reshape(labels.size, 1)
        return datax, labels

    def plot_data(self, ax, data_x, labels, plot_ideal_line=True, x_grid=None, y_grid=None, z_model=None,
                  add_bar=True):
        plt.sca(ax)
        plt.cla()

        xlims = np.array([self.min_x, self.max_x])
        ylims = np.array([self.min_x, self.max_x])

        if x_grid is not None and y_grid is not None and z_model is not None:
            CS = ax.contourf(x_grid, y_grid, z_model, np.linspace(0., 1., 51),
                             alpha=.8, cmap=plt.cm.get_cmap("bwr"), hold="on")
            decision_line = plt.contour(CS, levels=[0.5], hold="on", colors='k')
            if add_bar:
                cbar = plt.colorbar(CS)
                cbar.add_lines(decision_line)
                cbar.ax.set_ylabel('output value')

        if plot_ideal_line:
            x2 = -xlims*(self.a/self.b) - (self.c/self.b)
            plt.plot(xlims, x2, 'b', label='Ideal')

        ax.scatter(data_x[:, 0], data_x[:, 1], c=labels.flatten(), s=40)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.draw()


class ClassificationTwoSpiralsData:
    def __init__(self, degrees=570, start=40, min_x=-12, max_x=12):
        """
        Adopted from Matlab code at:
            http://www.mathworks.com/matlabcentral/fileexchange/
            41459-6-functions-for-generating-artificial-datasets/content/twospirals.m
        
        :type degrees: int
        :param degrees: length of the spirals

        :type start: int
        :param start: how far from the origin the spirals start, in degrees
        
        :param min_x: 
        :param max_x: 
        """
        self.degrees = degrees
        self.start = start
        self.min_x = min_x
        self.max_x = max_x

    def generate_data(self, n_points=2000, noise=0.5, seed=1234):
        """
        :type n_points: int
        :param n_points: number of instances
        :type noise: float
        :param noise: 0 is no noise, at 1 the spirals will start to overlap
        """

        deg2rad = 2.*math.pi/360.
        start = self.start*deg2rad

        N1 = int(math.floor(n_points / 2))
        N2 = n_points - N1

        rng = np.random.RandomState(seed)

        n = start + np.sqrt(rng.rand(N1)) * self.degrees * deg2rad
        d1 = np.column_stack((-np.cos(n)*n + rng.rand(N1)*noise,
                              np.sin(n)*n + rng.rand(N1)*noise, np.zeros_like(n)))

        n = start + np.sqrt(rng.rand(N2)) * self.degrees * deg2rad
        d2 = np.column_stack((np.cos(n)*n + rng.rand(N2)*noise,
                              -np.sin(n)*n + rng.rand(N2)*noise, np.ones_like(n)))

        d = np.vstack((d1, d2))

        order = np.arange(len(d))
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
        d = d[order]
        data_x = d[:, 0:2]
        data_labels = d[:, 2]
        data_labels = data_labels.reshape(data_labels.size, 1)

        return data_x, data_labels

    def plot_data(self, ax, data_x, labels, x_grid=None, y_grid=None, z_model=None,
                  add_bar=True, title=""):
        plt.sca(ax)
        plt.cla()

        xlims = np.array([self.min_x, self.max_x])
        ylims = np.array([self.min_x, self.max_x])

        if x_grid is not None and y_grid is not None and z_model is not None:
            CS = ax.contourf(x_grid, y_grid, z_model, np.linspace(0., 1., 51),
                             alpha=.8, cmap=plt.cm.get_cmap("bwr"), hold="on")
            decision_line = plt.contour(CS, levels=[0.5], hold="on", colors='k')
            if add_bar:
                cbar = plt.colorbar(CS)
                cbar.add_lines(decision_line)
                cbar.ax.set_ylabel('output value')

        ax.scatter(data_x[:, 0], data_x[:, 1], c=labels.flatten(), s=40)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.draw()


def input_noise(datax, noise_level=0.01):
    noisy_datax = datax + noise_level * np.random.randn(datax.size).reshape(datax.shape)
    return noisy_datax


def label_noise(labels, noise_level=0.05):
    r = np.random.rand(labels.size).reshape(labels.shape) < noise_level
    noisy_labels = np.logical_xor(labels, r).astype('float32')
    return noisy_labels


def prepare_grid_points_for_contour_plot(min_x, max_x, grid_size):
    values = np.linspace(min_x, max_x, grid_size)
    X1, X2 = np.meshgrid(values, values)
    X1vec = X1.reshape((X1.size, 1))
    X2vec = X2.reshape((X2.size, 1))
    return X1, X2, np.hstack((X1vec, X2vec))
