import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from datasets.dataset import Dataset
from losses.loss_function import Loss
from models import linear


def draw_data_points(dataset: Dataset, figure: Figure) -> None:
    data = [go.Scatter(x=dataset.x, y=dataset.y, mode="markers", line=dict(color="steelblue")),
            go.Scatter()]
    figure.add_traces(
        data=data,
        rows=[1] * len(data),
        cols=[1] * len(data)
    )
    figure.update_xaxes(range=[0, max(dataset.x)], title=dict(text=dataset.conf.x_label), row=1, col=1)
    figure.update_yaxes(range=[0, max(dataset.y)], title=dict(text=dataset.conf.y_label), row=1, col=1)


def draw_loss_function(loss_function: Loss, figure: Figure) -> None:
    data = [go.Scatter(x=loss_function.a, y=loss_function.loss, mode="lines", line=dict(color="steelblue"))] + \
           [go.Scatter()] * 2
    figure.add_traces(
        data=data,
        rows=[1] * len(data),
        cols=[2] * len(data)
    )
    figure.update_xaxes(range=[np.min(loss_function.a), np.max(loss_function.a)], title=dict(text='a'), row=1, col=2)
    figure.update_yaxes(range=[0, min(loss_function.loss[0], loss_function.loss[-1])],
                        title=dict(text='J(a)'), row=1, col=2)


def prepare_frame(a: float, loss: float) -> dict:
    rng = np.array([0, 100])
    data = [
        go.Scatter(x=rng, y=linear.model(a, rng), mode="lines", line=dict(color="tomato")),
        go.Scatter(x=[a, a], y=[0, loss], mode="lines+markers", line=dict(color="tomato", width=2, dash='dash')),
        go.Scatter(x=[0, a], y=[loss, loss], mode="lines+markers", line=dict(color="tomato", width=2, dash='dash'))
    ]
    return dict(
        data=data,
        traces=[1, 3, 4]
    )
