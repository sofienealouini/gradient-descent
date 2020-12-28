import numpy as np

from datasets.dataset import Dataset


class Loss:

    def __init__(self, dataset: Dataset, use_intercept: bool):
        self.dataset = dataset
        self.use_intercept = use_intercept
        self.a, self.b, self.loss = self.points()

    def points(self, number: int = 100):
        if not self.use_intercept:
            a = np.linspace(0., 2 * self.dataset.conf.slope, number)
            loss = np.array([Loss.function(a_i, 0, self.dataset.x, self.dataset.y) for a_i in a])
            return a, None, loss
        else:
            raise NotImplementedError('use_intercept == True not implemented yet')

    @staticmethod
    def function(a: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
        return 0.5 * np.mean(np.square((a * x + b) - y))
