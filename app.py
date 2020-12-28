from plotly.subplots import make_subplots

from datasets.dataset import Dataset, DatasetConfig
from graphics.graphs import draw_data_points, draw_loss_function, prepare_frame
from losses.loss_function import Loss
from models import linear

if __name__ == '__main__':
    # Generate the dataset
    dataset = Dataset(conf=DatasetConfig.load('apartment_prices'))

    # Build theoretical loss function
    loss_function = Loss(dataset=dataset, use_intercept=False)

    # Train the model
    a_hist, loss_hist = linear.train(dataset, epochs=100, lr=0.0004, early_stopping_delta=100)

    fig = make_subplots(rows=1, cols=2)
    draw_data_points(dataset=dataset, figure=fig)
    draw_loss_function(loss_function=loss_function, figure=fig)
    fig.update(frames=[prepare_frame(a, loss) for a, loss in zip(a_hist, loss_hist)])
    fig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Train", method="animate", args=[None])])],
                      showlegend=False)
    fig.show()
