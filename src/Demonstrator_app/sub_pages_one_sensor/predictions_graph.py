###############################################################################
# Libraries
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from annotated_text import annotated_text


from metrics import rmse
from utils_streamlit_app import load_numpy
from utils_streamlit_app import load_experiment_config


#######################################################################
# Function(s)
#######################################################################
def load_experiment_results(experiment_path, sensor_selected):
    y_true_unormalized = load_numpy(f"{experiment_path}/y_true_local_{sensor_selected}_unormalized.npy")
    y_pred_local_unormalized = load_numpy(f"{experiment_path}/y_pred_local_{sensor_selected}_unormalized.npy")
    y_pred_fed_unormalized = load_numpy(f"{experiment_path}/y_pred_fed_{sensor_selected}_unormalized.npy")
    test_set = load_numpy(f"{experiment_path}/test_data_{sensor_selected}_unormalized.npy")
    return y_true_unormalized, y_pred_local_unormalized, y_pred_fed_unormalized, test_set


#######################################################################
    # Figure(s)
#######################################################################
def add_y_true(fig, df):
    fig.add_scatter(
        x=df['Time'],
        y=df['y_true'],
        mode='lines',
        marker=dict(color='black'),
        name='y_true'
    )


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


def add_vrectangle(fig, x0, x1):
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor='gray',
        opacity=0.2,
        line_width=0
    )


def update_xaxis_range(fig, range_xaxis):
    fig.update_xaxes(
        title='Time',
        tickformat='%H:%M',
        range=range_xaxis,
        dtick=3600000
    )


def create_fig(df, color_fed, color_local, rmse_fed, rmse_local):
    fig = px.line()
    fig.add_scatter(
        x=df['Time'],
        y=df["Window"],
        mode='lines',
        marker=dict(color='black'),
        name='y_true',
        showlegend=False)
    add_y_true(fig, df)

    add_y_pred_federation(fig, df, color_fed, rmse_fed)
    add_y_pred_federation_link_window(fig, df, color_fed)

    add_y_pred_local(fig, df, color_local, rmse_local)
    add_y_pred_local_link_window(fig, df, color_local)
    return fig


def update_axis(fig, range_x_axis, range_y_axis):
    update_xaxis_range(fig, range_x_axis)
    update_yaxis_range(fig, range_y_axis)


def update_layout_fig(fig, config, index, i):
    fig.update_layout(
        title=f"| Federation vs Local | Day: {index[i + config['window_size']].strftime('%Y-%m-%d')} | Time prediction: {int(config['prediction_horizon'] * 5 / 60)}h ({index[i + config['window_size']].strftime('%Hh%Mmin')} to {index[i + config['window_size'] + config['prediction_horizon']].strftime('%Hh%Mmin)')} | Step: {i} |",
        title_font=dict(size=25),
        legend=dict(
            title='Legends',
            font=dict(size=15),
            yanchor='bottom',
            xanchor='right'
        )
    )
    add_vrectangle(fig, index[i], index[i + config["window_size"] - 1])


#######################################################################
    # Dataframe
#######################################################################
def add_y_pred_fed_to_df(df, config, y_pred_fed):
    df['y_pred_federation'] = np.concatenate([np.repeat(np.nan, config["window_size"]).reshape(-1, 1), y_pred_fed])
    df["y_pred_federation_link_window"] = np.concatenate([np.repeat(np.nan, config["window_size"]).reshape(-1, 1), y_pred_fed])
    df["y_pred_federation_link_window"].at[config["window_size"] - 1] = df['Window'].iloc[config["window_size"] - 1]
    return df


def add_y_pred_local_to_df(df, config, y_pred_local):
    df['y_pred_local'] = np.concatenate([np.repeat(np.nan, config["window_size"]).reshape(-1, 1), y_pred_local])
    df["y_pred_local_link_window"] = np.concatenate([np.repeat(np.nan, config["window_size"]).reshape(-1, 1), y_pred_local])
    df["y_pred_local_link_window"].at[config["window_size"] - 1] = df['Window'].iloc[config["window_size"] - 1]
    return df


def create_dataframe(index, test_set, config, y_pred_fed, y_pred_local, i):
    df = pd.DataFrame({'Time': index[i:i + config["window_size"] + config["prediction_horizon"]], 'Traffic Flow': test_set[i:i + config["window_size"] + config["prediction_horizon"]].flatten()})
    df['Window'] = df['Traffic Flow'].where((df['Time'] >= index[i]) & (df['Time'] < index[i + config["window_size"]]))
    df['y_true'] = df['Traffic Flow'].where((df['Time'] >= index[i + config["window_size"] - 1]))

    df = add_y_pred_fed_to_df(df, config, y_pred_fed)
    df = add_y_pred_local_to_df(df, config, y_pred_local)
    return df


def graph_prediction(experiment_path, index, config, sensor_selected, i):
    y_true, y_pred_local, y_pred_fed, test_set = load_experiment_results(experiment_path, sensor_selected)

    rmse_fed = rmse(y_true[i, :].flatten(), y_pred_fed[i, :].flatten())
    rmse_local = rmse(y_true[i, :].flatten(), y_pred_local[i, :].flatten())

    color_fed, color_local = ("#00dd00", "#fe4269")

    df = create_dataframe(index, test_set, config, y_pred_fed[i, :], y_pred_local[i, :], i)

    fig = create_fig(df, color_fed, color_local, rmse_fed, rmse_local)

    min_yaxis = min(min(y_true.flatten()), min((min(y_pred_fed.flatten()), min(y_pred_local.flatten()))))
    max_yaxis = max((max(y_true.flatten()), max(y_pred_fed.flatten()), max(y_pred_local.flatten())))
    update_axis(fig, [index[i], index[i + config["window_size"] + config["prediction_horizon"]]],
                [min_yaxis, max_yaxis])

    update_layout_fig(fig, config, index, i)
    return fig


#######################################################################
    # Render
#######################################################################
def render_prediction_graph(experiment_path, sensor_selected):
    config = load_experiment_config(experiment_path)

    index = load_numpy(f"{experiment_path}/index_{sensor_selected}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')

    slider = st.slider('Select the step (a step equal 5min)?', 0, len(index) - config["prediction_horizon"] - config["window_size"] - 1, 0, key="MAP_and_Graph")

    fig_fed_local = graph_prediction(experiment_path, index, config, sensor_selected, slider)

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
        render_prediction_graph(path_experiment_selected, sensor_selected)
