import numpy as np
import ipywidgets as widgets 
from IPython.display import display 
import matplotlib.pyplot as plt
import tables
import pandas as pd

class AsmlWidget:
    """
    Description:
        widget for visualizing assimilation
    Attributes:
        asml_file_path: path to assimilation file
        hidden_file_path: path to true trajectory
        attractor_path: path to attractor points
        attractor: array of attractor points
        data: handle to assimilation file
    """
    def __init__(self, asml_file_path, dims=[0, 1], hidden_file_path=None, attractor_path=None, max_attractor_pts=5000):
        self.asml_file_path = asml_file_path
        self.dims = dims
        self.attractor_path = attractor_path
        self.hidden_file_path = hidden_file_path
        self.data = tables.open_file(self.asml_file_path, 'r')
        self.observation = np.array(self.data.root.observation.read().tolist())
        self.num_steps = len(self.observation)
        if hidden_file_path is not None:
            self.hidden_path = self.get_hidden_path()
            print(self.hidden_path.shape)
        if attractor_path is not None:
            self.attractor = self.get_attractor()
            if len(self.attractor) > max_attractor_pts:
                self.attractor = self.attractor[np.random.choice(len(self.attractor), size=max_attractor_pts, replace=False)]
        self.asml_step_slider = widgets.IntSlider(value=0, min=0, max=self.num_steps, step=1)
        widgets.interact(self.ensemble_plot, step=self.asml_step_slider)


    def ensemble_plot(self, step):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        posterior = self.get_ensemble(step)
        if self.attractor_path is not None:
            ax.scatter(self.attractor[:, self.dims[0]], self.attractor[:, self.dims[1]], c='green', s=100)
        ax.scatter(posterior[:, self.dims[0]], posterior[:, self.dims[1]], c='blue')
        ax.scatter(self.hidden_path[step, self.dims[0]], self.hidden_path[step, self.dims[1]], c='pink')

    def get_ensemble(self, step):
        return np.array(getattr(self.data.root.particles, 'time_' + str(step)).read().tolist())

    def get_attractor(self):
        return np.genfromtxt(self.attractor_path, delimiter=',')

    def get_hidden_path(self):
        return np.genfromtxt(self.hidden_file_path, delimiter=',')[:self.num_steps]
        