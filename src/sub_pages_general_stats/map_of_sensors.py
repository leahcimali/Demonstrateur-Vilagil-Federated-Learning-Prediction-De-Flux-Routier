import streamlit as st
from streamlit_folium import folium_static
import folium
from screeninfo import get_monitors
from annotated_text import annotated_text


from metrics import maape
from utils_streamlit_app import create_circle_precision_predict, get_color_fed_vs_local, selection_of_experiment, load_experiment_config, load_experiment_results
from utils_streamlit_app import load_numpy


#######################################################################
# Constant(s)
#######################################################################
# Predefine 14 locations for sensors
SEATTLE_ROADS = [
    [47.679470, -122.315626],
    [47.679441, -122.306665],
    [47.683058163418266, -122.30074031156877],
    [47.67941986097163, -122.29031294544225],
    [47.67578888921566, -122.30656814568495],
    [47.67575649888934, -122.29026613694701],
    [47.68307457244817, -122.29054200791231],
    [47.68300028244276, -122.3121427044953],
    [47.670728396123444, -122.31192781883172],
    [47.675825, -122.315658],
    [47.69132417321706, -122.31221442807933],
    [47.68645681961068, -122.30076590191602],
    [47.68304467808857, -122.27975989945097],
    [47.6974488132659, -122.29057907732675]
]

height = []
width = []
for m in get_monitors():
    height.append(m.height)
    width.append(m.width)

HEIGHT = min(height)
WIDTH = min(width)


#######################################################################
# Function(s)
#######################################################################
def add_sensors_to_map(sensors_loc, map_folium):
    for sensor in sensors_loc.keys():
        tooltip = f"Road: {sensor}"
        folium.Marker(location=sensors_loc[sensor], tooltip=tooltip, icon=folium.Icon(color="black")).add_to(map_folium)


def text_introduction_map():
    st.divider()
    annotated_text(
        "A higher percent indicates a better prediction. So, ",
        ("green", "", "#75ff5b"), " circle",
        " is better than the ",
        ("red", "", "#fe7597"), " one.")
    st.write("""
            **Exemple**: for a prediction at one hour (t+12), 98% means that\\
            in average on the 12 points predicted the model has an accuracy\\
            equal to 98%.
            """)
    st.divider()


def map_sensor_to_loc(mapping_sensor_with_nodes):
    sensors_loc = {}
    seattle_roads_crop_by_nb_sensor = [SEATTLE_ROADS[i] for i in range(len(mapping_sensor_with_nodes.keys()))]

    for sensor, locations in zip(mapping_sensor_with_nodes.keys(), seattle_roads_crop_by_nb_sensor):
        sensors_loc[sensor] = locations
    return sensors_loc


def calculate_contrary_maape(y_true, y_pred):
    maape_value = maape(y_true.flatten(), y_pred.flatten())
    return 1 - maape_value / 100.0


def add_metric_on_map(y_true, y_pred_local, y_pred_fed, coords, seattle_map_fed, seattle_map_local):
    maape_computed_fed = calculate_contrary_maape(y_true, y_pred_fed)
    maape_computed_local = calculate_contrary_maape(y_true, y_pred_local)

    color_fed, color_local = get_color_fed_vs_local(maape_computed_fed, maape_computed_local)

    create_circle_precision_predict(coords, maape_computed_fed, seattle_map_fed, color_fed)
    create_circle_precision_predict(coords, maape_computed_local, seattle_map_local, color_local)


def render_map(seattle_map_fed, seattle_map_local):
    st.header("MAP")
    col1, col2 = st.columns((0.5, 0.5), gap="small")
    with col1:
        col1.header('Federated model results')
        folium_static(seattle_map_fed, width=WIDTH / 2 - 300)  # To fix the overlapping effect (handmade solution)
    with col2:
        col2.header('Local models results')
        folium_static(seattle_map_local, width=WIDTH / 2 - 300)  # To fix the overlapping effect (handmade solution)


def plot_map(experiment_path, mapping_sensor_with_nodes):
    text_introduction_map()

    seattle_map_fed = folium.Map(location=[47.6776, -122.30064], zoom_start=15, zoomSnap=0.25)
    seattle_map_local = folium.Map(location=[47.67763, -122.30064], zoom_start=15, zoomSnap=0.25)

    sensors_loc = map_sensor_to_loc(mapping_sensor_with_nodes)

    add_sensors_to_map(sensors_loc, seattle_map_fed)
    add_sensors_to_map(sensors_loc, seattle_map_local)

    for sensor in mapping_sensor_with_nodes.keys():
        y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor]}.npy")
        y_pred_local = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor]}.npy")
        y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor]}.npy")
        add_metric_on_map(y_true, y_pred_local, y_pred_fed, sensors_loc[sensor], seattle_map_fed, seattle_map_local)

    seattle_map_fed.fit_bounds(seattle_map_fed.get_bounds(), padding=(30, 30))
    seattle_map_local.fit_bounds(seattle_map_local.get_bounds(), padding=(30, 30))

    render_map(seattle_map_fed, seattle_map_local)


#######################################################################
# Main
#######################################################################
def map_of_sensors():
    st.subheader("Map")
    st.write("""
                * On this page select one experiment.
                    * On the map, you will find the sensors location and their accuracy.
                """)
    st.divider()

    path_experiment_selected = selection_of_experiment()

    if (path_experiment_selected is not None):
        results = load_experiment_results(path_experiment_selected)
        config = load_experiment_config(path_experiment_selected)

        mapping_sensor_with_node = {}
        for node in results.keys():  # e.g. keys = ['0', '1', '2', ...]
            mapping_sensor_with_node[config["nodes_to_filter"][int(node)]] = node  # e.g. nodes_to_filter = [118, 261, 10, ...]

        plot_map(path_experiment_selected, mapping_sensor_with_node)
