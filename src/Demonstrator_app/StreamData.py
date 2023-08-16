import numpy as np
import pandas as pd
import streamlit as st

from utils_streamlit_app import get_color_fed_vs_local, style_dataframe

METRICS = ["RMSE", "MAE", "MAAPE", "Superior Pred %"]


class StreamData:
    def __init__(self, stream_data, config_stream_data, path_to_exp=None):
        super(StreamData, self).__init__()
        if path_to_exp is not None:
            self.path_to_exp = path_to_exp
        self.data = stream_data
        self.parameters = config_stream_data
        self.indexes = list(stream_data.keys())
        self.sensors = [config_stream_data["nodes_to_filter"][int(index)] for index in stream_data.keys()]
        self.sensors_name = config_stream_data["nodes_to_filter"]
        self.name = config_stream_data["save_model_path"]
        self.size = len(stream_data)

    def get_sensor_metric_unormalized_local_values(self, node, metric):
        return self.data[node]["local_only_unormalized"][metric]

    def get_sensor_metric_unormalized_federated_values(self, node, metric):
        return self.data[node]["Federated_unormalized"][metric]

    def get_sensor_metric_normalized_local_values(self, node, metric):
        return self.data[node]["local_only_normalized"][metric]

    def get_sensor_metric_normalized_federated_values(self, node, metric):
        return self.data[node]["Federated_normalized"][metric]

    def get_results_federated(self, normalized=False):
        version = "Federated_normalized" if normalized else "Federated_unormalized"
        return [
            self.data[sensor][version]
            for sensor in self.indexes
        ]

    def get_results_local(self, normalized=False):
        version = "local_only_normalized" if normalized else "local_only_unormalized"
        return [
            self.data[sensor][version]
            for sensor in self.indexes
        ]

    def get_sensors_name_better_in_federated(self, metric):
        sensors_name = []
        for sensor in self.indexes:
            if metric == "Superior Pred %":
                if self.data[sensor]["Federated_unormalized"][metric] >= self.data[sensor]["local_only_unormalized"][metric]:
                    sensors_name.append(self.sensors_name[int(sensor)])
            elif self.data[sensor]["Federated_unormalized"][metric] <= self.data[sensor]["local_only_unormalized"][metric]:
                sensors_name.append(self.sensors_name[int(sensor)])
        return sensors_name

    def get_sensors_federated_stats(self, metric=None, normalized=True):
        df = pd.DataFrame(self.get_results_federated(normalized), columns=METRICS)
        if metric is None:
            return df.describe().T.drop(columns="count", inplace=False, axis=1)
        else:
            return df.describe().T.loc[metric]["mean"].item()

    def get_sensors_local_stats(self, metric=None, normalized=True):
        df = pd.DataFrame(self.get_results_local(normalized), columns=METRICS)
        if metric is None:
            return df.describe().T.drop(columns="count", inplace=False, axis=1)
        else:
            return df.describe().T.loc[metric]["mean"].item()

    def get_nb_sensor_better_in_federation(self, metric):
        nb_sensor = 0
        for sensor in self.indexes:
            if metric == "Superior Pred %":
                if self.data[sensor]["Federated_unormalized"][metric] >= self.data[sensor]["local_only_unormalized"][metric]:
                    nb_sensor += 1
            elif self.data[sensor]["Federated_unormalized"][metric] <= self.data[sensor]["local_only_unormalized"][metric]:
                nb_sensor += 1
        return nb_sensor

    def compute_average_rate_of_change(self, sensors=None, normalized=False):
        if normalized:
            federated_ver = "Federated_normalized"
            local_ver = "local_only_normalized"
        else:
            federated_ver = "Federated_unormalized"
            local_ver = "local_only_unormalized"

        if sensors is None:
            sensors = self.indexes
        avg_rate_change = {}
        for metric in METRICS:
            for sensor in sensors:
                if metric not in avg_rate_change.keys():
                    avg_rate_change[metric] = 1 + ((self.data[sensor][federated_ver][metric] - self.data[sensor][local_ver][metric]) / self.data[sensor][local_ver][metric])
                else:
                    avg_rate_change[metric] = avg_rate_change[metric] * (1 + ((self.data[sensor][federated_ver][metric] - self.data[sensor][local_ver][metric]) / self.data[sensor][local_ver][metric]))
        for metric in METRICS:
            avg_rate_change[metric] = (np.power(avg_rate_change[metric], (1 / len(sensors))) - 1) * 100
        return avg_rate_change

    def show_results_sensor(self, sensor, normalized=False):
        if normalized:
            federated_ver = "Federated_normalized"
            local_ver = "local_only_normalized"
        else:
            federated_ver = "Federated_unormalized"
            local_ver = "local_only_unormalized"
        df_fed = pd.DataFrame(self.data[sensor][federated_ver], columns=METRICS, index=["Value"]).T
        df_fed = df_fed.applymap(lambda x: '{:.4f}'.format(x))
        df_local = pd.DataFrame(self.data[sensor][local_ver], columns=METRICS, index=["Value"]).T
        df_local = df_local.applymap(lambda x: '{:.4f}'.format(x))
        color_fed = []
        color_local = []
        for i in range(len(METRICS)):
            if (i < len(METRICS) - 1):
                col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i]["Value"], df_local.iloc[i]["Value"], superior=False)
            else:  # because "Superior Pred %" metric needs to be superior=True
                col_fed, col_local = get_color_fed_vs_local(df_fed.iloc[i]["Value"], df_local.iloc[i]["Value"], superior=True)
            color_fed.append(col_fed)
            color_local.append(col_local)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.subheader("Federated")
            # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
            st.table(df_fed.style.set_table_styles(style_dataframe(df_fed, colors=color_fed, column_index=2)))
        with c2:
            st.subheader("Local")
            # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
            st.table(df_local.style.set_table_styles(style_dataframe(df_local, colors=color_local, column_index=2)))

        avg_rate_change = self.compute_average_rate_of_change([sensor], normalized)
        st.subheader("Average rate of change Local to Federated version")
        avg_rate_change = pd.DataFrame.from_dict(avg_rate_change, orient="index", columns=["Average rate of change"])
        avg_rate_change = avg_rate_change.applymap(lambda x: '{:.4f} %'.format(x))
        st.table(avg_rate_change.style.set_table_styles(style_dataframe(avg_rate_change, colors="#000000", column_index=2)))

    def show_results(self, normalized=False):
        stats_fed_ver = self.get_sensors_federated_stats(normalized=normalized)
        stats_fed_ver = stats_fed_ver.applymap(lambda x: '{:.4f}'.format(x))
        stats_local_ver = self.get_sensors_local_stats(normalized=normalized)
        stats_local_ver = stats_local_ver.applymap(lambda x: '{:.4f}'.format(x))
        mean_stats_fed_ver = stats_fed_ver["mean"]
        mean_stats_local_ver = stats_local_ver["mean"]

        df_mean_diff = pd.DataFrame({"local": mean_stats_local_ver, "diff_on_mean": mean_stats_fed_ver})
        df_mean_diff["diff_on_mean"] = df_mean_diff["diff_on_mean"].astype(float)
        df_mean_diff["local"] = df_mean_diff["local"].astype(float)

        color_fed = []
        color_local = []
        for metric in METRICS:
            if (metric != "Superior Pred %"):
                col_fed, col_local = get_color_fed_vs_local(stats_fed_ver.loc[metric]["mean"], stats_local_ver.loc[metric]["mean"], superior=False)
            else:  # because "Superior Pred %" metric needs to be superior=True
                col_fed, col_local = get_color_fed_vs_local(stats_fed_ver.loc[metric]["mean"], stats_local_ver.loc[metric]["mean"], superior=True)
            color_fed.append(col_fed)
            color_local.append(col_local)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.subheader("Federated")
            # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
            st.table(stats_fed_ver.style.set_table_styles(style_dataframe(stats_fed_ver, colors=color_fed, column_index=2)))
        with c2:
            st.subheader("Local")
            # use st.table because st.dataframe is not personalizable for the moment (version 1.22)
            st.table(stats_local_ver.style.set_table_styles(style_dataframe(stats_local_ver, colors=color_local, column_index=2)))

        st.subheader("Difference between federated and local value on the mean")
        df_mean_diff = df_mean_diff.diff(axis=1)
        df_mean_diff.drop("local", axis=1, inplace=True)
        df_mean_diff = df_mean_diff.applymap(lambda x: '{:.4f}'.format(x))
        st.table(df_mean_diff.style.set_table_styles(style_dataframe(df_mean_diff, colors="#000000", column_index=2)))

        avg_rate_change = self.compute_average_rate_of_change(normalized=normalized)
        st.subheader("Average rate of change Local to Federated version")
        avg_rate_change = pd.DataFrame.from_dict(avg_rate_change, orient="index", columns=["Average rate of change"])
        avg_rate_change = avg_rate_change.applymap(lambda x: '{:.4f} %'.format(x))
        st.table(avg_rate_change.style.set_table_styles(style_dataframe(avg_rate_change, colors="#000000", column_index=2)))

    def show_parameters(self):
        df_parameters = pd.DataFrame(self.parameters, columns=["time_serie_percentage_length",
                                                    "batch_size",
                                                    "window_size",
                                                    "prediction_horizon",
                                                    "communication_rounds",
                                                    "num_epochs_local_federation",
                                                    "num_epochs_local_no_federation",
                                                    "model"], index=["Value"])
        column_names = {
            "time_serie_percentage_length": "Length of the time serie used",
            "batch_size": "Batch Size",
            "window_size": "WS",
            "prediction_horizon": "PH",
            "communication_rounds": "CR",
            "num_epochs_local_no_federation": "Epochs alone",
            "num_epochs_local_federation": "Epochs Federation",
            "learning_rate": "Learning Rate",
            "model": "Model"
        }
        st.subheader("Parameters of the experiments")
        st.write("")
        st.write("Length of Time Series: Percentage of the time series used before splitting the dataset into train/validation/test sets.")
        st.write("Window Size (**WS**): The number of time steps in the historical data considered by the model for making predictions.")
        st.write("Prediction Horizon (**PH**): The number of time steps or observations to forecast beyond the last observation in the input window.")
        st.write("Communication Round (**CR**): The iteration or cycle of communication between the central server and the actors during the training process.")
        st.write("Epochs Federation: The number of epochs an actor performs before sending its model to the central server.")
        st.write("Epochs local alone: The number of epochs used to train the local version, which will be compared to the federated version.")

        df_parameters.index.name = "Parameters"
        df_parameters = df_parameters.rename(columns=column_names)
        df_parameters["WS"] = pd.Series(df_parameters["WS"]).apply(lambda x: f"t+{x} ({int((float(x) * 5) / 60)}h)")[0]
        df_parameters["PH"] = pd.Series(df_parameters["PH"]).apply(lambda x: f"t+{x} ({int((float(x) * 5) / 60)}h)")[0]
        st.dataframe(df_parameters.T, use_container_width=True)
