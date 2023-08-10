###############################################################################
# Libraries
###############################################################################
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from config import Params
from annotated_text import annotated_text


from metrics import rmse
from utils_streamlit_app import get_color_fed_vs_local, load_numpy


###############################################################################
# Function(s)
###############################################################################
def remove_outliers(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]


def load_experiment_results(experiment_path, sensor_selected):
    y_true_unormalized = load_numpy(f"{experiment_path}/y_true_local_{sensor_selected}_unormalized.npy")
    y_pred_local_unormalized = load_numpy(f"{experiment_path}/y_pred_local_{sensor_selected}_unormalized.npy")
    y_pred_fed_unormalized = load_numpy(f"{experiment_path}/y_pred_fed_{sensor_selected}_unormalized.npy")
    return y_true_unormalized, y_pred_local_unormalized, y_pred_fed_unormalized


def text_introduction_map():
    annotated_text(("green", "", "#75ff5b"), " means this the version with the lower absolute error in average",
        " and the ",
        ("red", "", "#fe7597"), " means the opposite.")
    st.divider()


def plot_box_configurable(title, ae, max_y_value, color, params, sensor_selected):
    fig = go.Figure()
    box = go.Box(y=ae, marker_color=color, boxmean='sd', name=title, boxpoints=False)
    fig.add_trace(box)
    fig.update_layout(
        title={
            'text': f"{title}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=f"sensor_select {params.nodes_to_filter[int(sensor_selected)]}",
        yaxis_title="Trafic flow (absolute error)",
        yaxis=dict(range=[0, max_y_value]),
        font=dict(
            size=28,
            color="#FF7f7f"
        ),
        height=900, width=350
    )
    fig.update_traces(jitter=0)
    return fig


def render_boxplot(ae_fed, ae_local, max_ae, color_fed, color_local, params, sensor_selected):
    # FEDERATED
    fed_fig = plot_box_configurable("Federated Prediction", ae_fed, max_ae, color_fed, params, sensor_selected)
    # LOCAL
    local_fig = plot_box_configurable("Local Prediction", ae_local, max_ae, color_local, params, sensor_selected)

    with st.spinner('Plotting...'):
        st.subheader(f"Comparison between federated and local version on sensor {params.nodes_to_filter[int(sensor_selected)]} (Absolute Error)")
        _, c2_fed_fig, c3_local_fig, _ = st.columns((1, 1, 1, 1))
        with c2_fed_fig:
            st.plotly_chart(fed_fig, use_container_width=False)
        with c3_local_fig:
            st.plotly_chart(local_fig, use_container_width=False)


def plot_box(experiment_path, sensor_selected):
    params = Params(f'{experiment_path}/config.json')
    index = load_numpy(f"{experiment_path}/index_{sensor_selected}.npy")
    st.markdown(":red[THE SLIDER IS DISABLED FOR THIS VISUALIZATION]")
    st.slider('Select the step (a step equal 5min)?', 0, len(index) - params.prediction_horizon - params.window_size - 1, 0, key="MAP_and_Graph", disabled=True)

    text_introduction_map()

    y_true, y_pred_local, y_pred_fed = load_experiment_results(experiment_path, sensor_selected)

    ae_fed = remove_outliers((np.abs(y_pred_fed.flatten() - y_true.flatten())))
    ae_local = remove_outliers((np.abs(y_pred_local.flatten() - y_true.flatten())))
    max_ae = max(max(ae_fed), max(ae_local))

    rmse_local = rmse(y_true.flatten(), y_pred_local.flatten())
    rmse_fed = rmse(y_true.flatten(), y_pred_fed.flatten())

    color_fed, color_local = get_color_fed_vs_local(rmse_fed, rmse_local, superior=False)

    render_boxplot(ae_fed, ae_local, max_ae, color_fed, color_local, params, sensor_selected)


#######################################################################
# Main
#######################################################################
def box_plot_sensor(path_experiment_selected, sensor_selected):
    st.subheader("BoxPlot")
    st.write("""
            * On the boxplots, you will see the distribution of the absolute error for the selected sensor.
                * The left boxplot represents the federated version, while the right boxplot represents the local version.
            """)
    st.divider()

    if (path_experiment_selected is not None):
        plot_box(path_experiment_selected, sensor_selected)
