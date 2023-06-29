###############################################################################
# Libraries
###############################################################################
from os import path


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from annotated_text import annotated_text


from metrics import rmse
from config import Params
from utils_streamlit_app import load_numpy
from utils_streamlit_app import get_color_fed_vs_local


#######################################################################
# Function(s)
#######################################################################
def plot_prediction_graph(experiment_path, sensor_selected):
    annotated_text(
        "A lower RMSE value indicates a better prediction. The ",
        ("green", "", "#75ff5b"), " prediction",
        " is better than the ",
        ("red", "", "#fe7597"), " one because it has a lower RMSE value")
    st.divider()

    params = Params(f"{experiment_path}/config.json")
    test_set = load_numpy(f"{experiment_path}/test_data_{sensor_selected}.npy")

    y_true = load_numpy(f"{experiment_path}/y_true_local_{sensor_selected}.npy")
    y_pred_local = load_numpy(f"{experiment_path}/y_pred_local_{sensor_selected}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{sensor_selected}.npy")

    index = load_numpy(f"{experiment_path}/index_{sensor_selected}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')

    slider = st.slider('Select the step (a step equal 5min)?', 0, len(index) - params.prediction_horizon - params.window_size - 1, 0, key="MAP_and_Graph")

    def plot_graph_slider(y_pred_fed, y_pred_local, i):
        rmse_fed = rmse(y_true[slider, :].flatten(), y_pred_fed[slider, :].flatten())
        rmse_local = rmse(y_true[slider, :].flatten(), y_pred_local[slider, :].flatten())

        color_fed, color_local = get_color_fed_vs_local(rmse_fed, rmse_local, superior=False)

        df = pd.DataFrame({'Time': index[i:i + params.window_size + params.prediction_horizon], 'Traffic Flow': test_set[i:i + params.window_size + params.prediction_horizon].flatten()})
        df['Window'] = df['Traffic Flow'].where((df['Time'] >= index[i]) & (df['Time'] < index[i + params.window_size]))
        df['y_true'] = df['Traffic Flow'].where((df['Time'] >= index[i + params.window_size - 1]))

        df['y_pred_federation'] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred_fed[i, :]])
        df["y_pred_federation_link_window"] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred_fed[i]])
        df["y_pred_federation_link_window"].at[params.window_size - 1] = df['Window'].iloc[params.window_size - 1]

        df['y_pred_local'] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred_local[i, :]])
        df["y_pred_local_link_window"] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred_local[i]])
        df["y_pred_local_link_window"].at[params.window_size - 1] = df['Window'].iloc[params.window_size - 1]

        fig = px.line(
            df, x='Time',
            y=["Window"],
            color_discrete_sequence=['black']
        )
        fig.add_scatter(
            x=df['Time'],
            y=df['y_true'],
            mode='lines',
            marker=dict(color='blue'),
            name='y_true'
        )
        fig.add_scatter(
            x=df['Time'],
            y=df['y_pred_federation'],
            mode='markers+lines',
            marker=dict(color=color_fed, symbol="x", size=7),
            name=f'Federation RMSE : {rmse_fed:.2f}'
        )
        fig.add_scatter(
            x=df['Time'],
            y=df["y_pred_federation_link_window"],
            mode='lines',
            marker=dict(color=color_fed),
            showlegend=False
        )
        fig.add_scatter(
            x=df['Time'],
            y=df['y_pred_local'],
            mode='markers+lines',
            marker=dict(color=color_local, symbol="circle-open", size=7),
            name=f'Local RMSE : {rmse_local:.2f}'
        )
        fig.add_scatter(
            x=df['Time'],
            y=df["y_pred_local_link_window"],
            mode='lines',
            marker=dict(color=color_local),
            showlegend=False
        )
        fig.add_vrect(
            x0=index[i],
            x1=index[i + params.window_size - 1],
            fillcolor='gray',
            opacity=0.2,
            line_width=0
        )
        fig.update_xaxes(
            title='Time',
            tickformat='%H:%M',
            range=(index[i], index[i + params.window_size + params.prediction_horizon]),
            dtick=3600000
        )
        fig.update_yaxes(
            title="Traffic Flow",
            range=[min(min(y_true.flatten()), min((min(y_pred_fed.flatten()), min(y_pred_local.flatten())))),
                    max((max(y_true.flatten()), max(y_pred_fed.flatten()), max(y_pred_local.flatten())))],
            dtick=50
        )
        fig.update_layout(
            title=f"| Federation vs Local | Day: {index[slider + params.window_size].strftime('%Y-%m-%d')} | Time prediction: {int(params.prediction_horizon * 5 / 60)}h ({index[slider + params.window_size].strftime('%Hh%Mmin')} to {index[slider + params.window_size + params.prediction_horizon].strftime('%Hh%Mmin)')} | Step: {slider} |",
            title_font=dict(size=25),
            legend=dict(title='Legends', font=dict(size=16)),
        )
        fig.update_layout(
            legend=dict(
                title='Legends',
                font=dict(size=15),
                yanchor='bottom',
                xanchor='right'
            )
        )
        return fig

    st.subheader(f"working on sensor {params.nodes_to_filter[int(sensor_selected)]}")

    fig_fed_local = plot_graph_slider(y_pred_fed, y_pred_local, slider)

    with st.spinner('Plotting...'):
        st.plotly_chart(fig_fed_local, use_container_width=True)


#######################################################################
# Main
#######################################################################
def prediction_graph_sensor(path_experiment_selected, sensor_selected):
    st.subheader("Predictions Graph")
    st.write("""
            * On this page select two experiments to compare them.
                * In the table, you will find the general statistics for both the Local version and\\
                the Federated version on differents metrics. On the left the left model and on the\\
                right the other model.
                * In the box plot, you will see the distribution of the RMSE values.
            """)
    st.divider()

    if (path_experiment_selected is not None):
        if (path.exists(f'{path_experiment_selected}/y_true_local_{sensor_selected}.npy') and
            path.exists(f"{path_experiment_selected}/y_pred_fed_{sensor_selected}.npy")):
            plot_prediction_graph(path_experiment_selected, sensor_selected)

