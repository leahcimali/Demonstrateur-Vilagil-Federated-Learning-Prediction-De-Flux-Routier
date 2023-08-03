import sys

sys.path.append("./src/")

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QWheelEvent
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QSizePolicy, QSlider, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView
from pathlib import PurePath
from metrics import rmse
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import pandas as pd
import pyqtgraph as pg


from src.ClusterData import ClusterData
from src.utils_data import load_PeMS04_flow_data
from src.utils_graph import create_graph
from src.utils_streamlit_app import load_experiment_results, load_numpy, get_color_fed_vs_local, load_experiment_config


def load_experiment_results_sensor(experiment_path, sensor_selected):
    y_true_unormalized = load_numpy(f"{experiment_path}/y_true_local_{sensor_selected}_unormalized.npy")
    y_pred_local_unormalized = load_numpy(f"{experiment_path}/y_pred_local_{sensor_selected}_unormalized.npy")
    y_pred_fed_unormalized = load_numpy(f"{experiment_path}/y_pred_fed_{sensor_selected}_unormalized.npy")
    test_set = load_numpy(f"{experiment_path}/test_data_{sensor_selected}_unormalized.npy")
    return y_true_unormalized, y_pred_local_unormalized, y_pred_fed_unormalized, test_set

#######################################################################
    # Figure(s)
#######################################################################
def add_y_pred_federation(fig, df, color_fed, rmse_fed):
    fig.add_scatter(
        x=df['Time'],
        y=df['y_pred_federation'],
        mode='markers+lines',
        marker=dict(color=color_fed, symbol="x", size=7),
        name=f'Federation RMSE : {rmse_fed:.2f}'
    )


def add_y_pred_federation_link_window(fig, df, color_fed):
    fig.add_scatter(
        x=df['Time'],
        y=df["y_pred_federation_link_window"],
        mode='lines',
        marker=dict(color=color_fed),
        showlegend=False
    )


def add_y_pred_local(fig, df, color_local, rmse_local):
    fig.add_scatter(
        x=df['Time'],
        y=df['y_pred_local'],
        mode='markers+lines',
        marker=dict(color=color_local, symbol="circle-open", size=7),
        name=f'Local RMSE : {rmse_local:.2f}'
    )


def add_y_pred_local_link_window(fig, df, color_local):
    fig.add_scatter(
        x=df['Time'],
        y=df["y_pred_local_link_window"],
        mode='lines',
        marker=dict(color=color_local),
        showlegend=False
    )


def update_yaxis_range(fig, range_yaxis):
    fig.update_yaxes(
        title="Traffic Flow",
        range=range_yaxis,
        dtick=50
    )


def update_xaxis_range(fig, range_xaxis):
    fig.update_xaxes(
        title='Time',
        tickformat='%H:%M',
        range=range_xaxis,
        dtick=300000
    )


def create_fig(df, color_fed, color_local, rmse_fed, rmse_local):
    fig = go.Figure()

    add_y_pred_federation(fig, df, color_fed, rmse_fed)
    # add_y_pred_federation_link_window(fig, df, color_fed)

    # add_y_pred_local(fig, df, color_local, rmse_local)
    # add_y_pred_local_link_window(fig, df, color_local)
    return fig


def update_axis(fig, range_x_axis, range_y_axis):
    update_xaxis_range(fig, range_x_axis)
    update_yaxis_range(fig, range_y_axis)


def update_layout_fig(fig):
    fig.update_layout(
        title="",
        title_font=dict(size=25),
        legend=dict(
            title='Legends',
            font=dict(size=15),
            yanchor='bottom',
            xanchor='right'
        )
    )


#######################################################################
    # Dataframe
#######################################################################
def add_y_pred_fed_to_df(df, config, y_pred_fed):
    df['y_pred_federation'] = y_pred_fed
    return df


def create_dataframe(index, test_set, config, y_pred_fed, y_pred_local, i):
    df = pd.DataFrame({'Time': index[i + config["window_size"]: i + config["window_size"] + config["prediction_horizon"]], 'Traffic Flow': test_set[i + config["window_size"]: i + config["window_size"] + config["prediction_horizon"]].flatten()})

    df = add_y_pred_fed_to_df(df, config, y_pred_fed)
    return df


def graph_prediction(experiment_path, index, config, sensor_selected, i=0):
    y_true, y_pred_local, y_pred_fed, test_set = load_experiment_results_sensor(experiment_path, sensor_selected)

    rmse_fed = rmse(y_true[i, :].flatten(), y_pred_fed[i, :].flatten())
    rmse_local = rmse(y_true[i, :].flatten(), y_pred_local[i, :].flatten())

    color_fed, color_local = ("#00dd00", "#fe4269")

    df = create_dataframe(index, test_set, config, y_pred_fed[i, :], y_pred_local[i, :], i)

    fig = pg.PlotWidget()  # create_fig(df, color_fed, color_local, rmse_fed, rmse_local)
    fig.plot(df["Time"], df["y_pred_federation"], pen=pg.mkPen(color='b', width=2), symbol='o', symbolPen='r', symbolBrush='g')

    # min_yaxis = min(min(y_true.flatten()), min((min(y_pred_fed.flatten()), min(y_pred_local.flatten()))))
    # max_yaxis = max((max(y_true.flatten()), max(y_pred_fed.flatten()), max(y_pred_local.flatten())))
    # update_axis(fig, [index[i + config["window_size"]], index[i + config["window_size"] + config["prediction_horizon"]]],
    #             [min_yaxis, max_yaxis])

    # update_layout_fig(fig)
    return fig


class ZoomableWebEngineView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0

    def wheelEvent(self, event: QWheelEvent):
        # Handle zoom in/out using the mouse wheel
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor *= 0.9
        self.setZoomFactor(self.scale_factor)


class MainWindow(QMainWindow):
    def __init__(self, G, cluster):
        super().__init__()
        self.cluster = cluster
        self.index = load_numpy(f"{cluster.path_to_exp}/index_{cluster.indexes[0]}.npy")
        self.index = pd.to_datetime(self.index, format='%Y-%m-%dT%H:%M:%S.%f')

        # Create QGraphicsView to hold the plot widgets
        self.plot_view = QGraphicsView(self)
        self.plot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.plot_view)

        # Create QGraphicsScene to hold the plot widgets
        self.plot_scene = QGraphicsScene(self)
        self.plot_scene.setSceneRect(0, 0, 2450, 650)  # Prendre une plus grande taille pour accueillir tous les graphiques (widget)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(len(self.index))
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.update_slider)

        widget_in_scene = self.plot_scene.addWidget(self.slider)
        widget_in_scene.setPos(50, 650)

        self.button = QPushButton("Start Slider")
        self.button.clicked.connect(self.toggle_slider)
        widget_in_scene = self.plot_scene.addWidget(self.button)
        widget_in_scene.setPos(50, 700)

        self.slider_timer = None
        self.slider_direction = 1

        self.map_sensor_graph = {}
        self.map_sensor_webview = {}
        for sensor in ['0', '1', '2']:
            # self.map_sensor_webview[sensor] = QWebEngineView()
            self.map_sensor_graph[sensor] = graph_prediction(cluster.path_to_exp, self.index, cluster.parameters, sensor, 0)
            # self.map_sensor_webview[sensor].setHtml(self.map_sensor_graph[sensor].to_html(include_plotlyjs='cdn'))
            # self.map_sensor_webview[sensor].setContentsMargins(0, 0, 0, 0)
            # self.map_sensor_webview[sensor].setFixedSize(800, 600)
            widget_in_scene = self.plot_scene.addWidget(self.map_sensor_graph[sensor])
            widget_in_scene.setPos(805 * int(sensor), 0)

        # Add the QGraphicsScene to the QGraphicsView
        self.plot_view.setScene(self.plot_scene)

        # Enable scrolling and zooming in the QGraphicsView
        self.plot_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.plot_view.setVerticalScrollBarPolicy(0)
        self.plot_view.setHorizontalScrollBarPolicy(0)
        self.plot_view.setInteractive(True)
        self.plot_view.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.plot_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.plot_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.plot_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Set the main window size
        self.setGeometry(100, 100, 800, 600)  # Taille de la fenêtre de l'application

    # Implement zoom functionality for the plot widgets (you can adjust the scale factor as needed)
    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            factor = 1.2
            if event.angleDelta().y() > 0:
                self.plot_view.scale(factor, factor)
            else:
                self.plot_view.scale(1 / factor, 1 / factor)
        else:
            super().wheelEvent(event)

    def update_slider(self, value):
        self.plot_scene.setSceneRect(0, 0, 2450, 650)  # Prendre une plus grande taille pour accueillir tous les graphiques (widget)
        for sensor in ['0', '1', '2']:
            y_true, y_pred_local, y_pred_fed, test_set = load_experiment_results_sensor(self.cluster.path_to_exp, sensor)
            df = create_dataframe(self.index, test_set, self.cluster.parameters, y_pred_fed[value, :], y_pred_local[value, :], value)
            self.map_sensor_graph[sensor].clear()
            self.map_sensor_graph[sensor].plot(df["Time"], df["y_pred_federation"], pen='r')

    def toggle_slider(self):
        if self.slider_timer is None:
            self.slider_timer = self.startTimer(500)  # Adjust the timer delay as needed
            self.button.setText("Stop Slider")
        else:
            self.killTimer(self.slider_timer)
            self.slider_timer = None
            self.button.setText("Start Slider")

    def timerEvent(self, event):
        current_value = self.slider.value()
        if self.slider_direction == 1:
            if current_value < self.slider.maximum():
                self.slider.setValue(current_value + 1)
            else:
                self.slider_direction = -1
        else:
            if current_value > self.slider.minimum():
                self.slider.setValue(current_value - 1)
            else:
                self.slider_direction = 1


if __name__ == '__main__':
    path_exp = "community_experiments/28_cluster_with_Louvain_Algo/community0/"
    clusters = {}
    cluster = ClusterData(load_experiment_results(path_exp), load_experiment_config(path_exp), path_exp)

    _, distance = load_PeMS04_flow_data()
    G = create_graph(distance)
    app = QApplication(sys.argv)
    window = MainWindow(G, cluster)
    window.show()
    sys.exit(app.exec_())
