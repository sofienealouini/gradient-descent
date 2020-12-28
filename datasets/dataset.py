from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from config import DATASETS_CONFIG_PATH


@dataclass
class DatasetConfig:
    n_points: int
    slope: float
    intercept: float
    x_label: str
    x_min: float
    x_max: float
    y_label: str
    y_positive_only: bool
    y_noise_standard_deviation: float

    @staticmethod
    def load(dataset_name: str) -> 'DatasetConfig':
        with open(DATASETS_CONFIG_PATH) as file:
            return DatasetConfig(**yaml.load(file, Loader=yaml.FullLoader)[dataset_name])


class Dataset:

    def __init__(self, conf: DatasetConfig):
        self.conf: DatasetConfig = conf
        self.points = self.build()
        self.x = self.points[self.conf.x_label]
        self.y = self.points[self.conf.y_label]
        self.size = len(self.points)

    def build(self) -> pd.DataFrame:
        x = np.around(np.random.uniform(self.conf.x_min, self.conf.x_max, self.conf.n_points), decimals=2)
        y_noise = np.random.normal(loc=0., scale=self.conf.y_noise_standard_deviation, size=self.conf.n_points)
        y = np.around(self.conf.slope * x + self.conf.intercept + y_noise, decimals=2)
        if self.conf.y_positive_only:
            y = np.maximum(0., y)
        return pd.DataFrame(data={self.conf.x_label: x, self.conf.y_label: y})

    def to_csv(self, path: str) -> None:
        self.points.to_csv(path, header=True, index=False)
