

class Params():

    def __init__(self, config_file_path):

        import json

        # Load the configuration file using the provided path
        with open(config_file_path) as f:
            config = json.load(f)
            self.time_serie_percentage_length = config["time_serie_percentage_length"]
            self.batch_size = config["batch_size"]
            self.init_node = config['init_node']
            self.n_neighbours = config['n_neighbours']
            self.smooth = config['smooth']
            self.center_and_reduce = config['center_and_reduce']
            self.normalize = config['normalize']
            self.sort_by_mean = config['sort_by_mean']
            self.nodes_to_filter = config['nodes_to_filter']
            self.number_of_nodes = config['number_of_nodes']
            self.window_size = config['window_size']
            self.prediction_horizon = config['prediction_horizon']
            self.stride = config['stride']
            self.communication_rounds = config['communication_rounds']
            self.num_epochs_local_no_federation = config['num_epochs_local_no_federation']
            self.num_epochs_local_federation = config['num_epochs_local_federation']
            self.epoch_local_retrain_after_federation = config["epoch_local_retrain_after_federation"]
            self.learning_rate = config['learning_rate']
            self.model = config["model"]
            self.save_model_path = config["save_model_path"]
            self.model_input_size = config["model_input_size"]
            self.model_hidden_size = config["model_hidden_size"]
            self.model_num_layers = config["model_num_layers"]
            self.model_output_size = config["model_output_size"]


def convert_PeMS_to_csv(flow_file="./../data/PEMS04/PEMS04.npz", csv_file="./../data/PEMS04/distance.csv"):

    """
    Small utils function to convert a Npz file for the PeMS data into a csv
    """

    import numpy as np
    import pandas as pd

    data = np.load(flow_file)

    array_flow = data['data']
    array_flow = array_flow[:, :, 0]

    flow_dict = {k: array_flow[:, k] for k in range(array_flow.shape[1])}
    df_PeMS = pd.DataFrame(flow_dict)
    start_date = "2018-01-01 00:00:00"
    end_date = "2018-02-28 23:55:00"
    interval = "5min"
    index = pd.date_range(start=start_date, end=end_date, freq=interval)
    df_PeMS = df_PeMS.set_index(index)
    df_PeMS.to_csv(csv_file)


class Tee:
    # Class so that the stdout output is saved in a txt file and print at the same time
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()  # Flush the buffer to ensure immediate writing

    def flush(self):
        for file in self.files:
            file.flush()
