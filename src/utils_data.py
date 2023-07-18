import src.utils_graph as gu
from pathlib import Path
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exp_smooth(df_PeMS, alpha=0.2):
    
    """
    Simple Exponential smoothing using the Holt Winters method without using statsmodel
    
    Parameters:
    -----------
    df_PeMS : pd.DataFrame 
        data to smooth
    alpha : float
        exponential smoothing param

    Returns
    -------
    pd.Dataframe
        Dataframe with the input smoothed
    """

    df_PeMS = df_PeMS.ewm(alpha=alpha).mean()

    return df_PeMS


def normalize_data(df_PeMS):
    """
    Normalize the data diving by the maximum to put it between 0 and 1
    
    Parameters:
    -----------
    df_PeMs : pd.DataFrame 
        data to normalize

    Returns
    -------
    pd.Dataframe
        Dataframe with the input normalized
    """
    maximum = df_PeMS.max().max()
    df_PeMS = df_PeMS /  maximum
    return df_PeMS

def center_reduce(df):
    
    """
    Center and reduce the data to put it between -1 and 1 with mean 0 and std 1
    
    Parameters:
    -----------
    df : pd.DataFrame 
        data to center and reduce

    Returns
    -------
    normalized_df : pd.Dataframe
        Dataframe with the input center and reduce
    meanstd_dict : dictionary
        Dictionary containing the mean and std of all columns to unormalize data
    """
   
    meanstd_dict={}
    for column in df.columns:
        colmean = df[column].mean()
        colstd = df[column].std()
        meanstd_dict[column] = {'mean':colmean,'std':colstd}
    
    normalized_df=(df-df.mean())/df.std()

    return normalized_df, meanstd_dict

def createExperimentsData(cluster_size, df_PeMS, layers = 6, perc_train = 0.7, perc_val = 0.15, subgraph = False, overwrite = False):
    import pickle 
    from src.models import LSTMModel

    """
    Generates pickled (.pkl) dictionary files with the train/val/test data and an associated model

    Parameters
    ----------
    cluster_size : int
        Size of the node clusters

    df_PeMs : pd.Dataframe
        dataframe with all the PeMS data 

    layers: int
        number of layers for the NN model

    perc_train : float
        percentage of the data to be used for training

    perc_test : float
        percentage of the data to be used for testing

    """
    
    train_len = len(df_PeMS)

    if subgraph:
        dirpath = './experiment/cluster'
        subgraph = gu.subgraph_dijkstra(G,i, cluster_size-1)
        nodes_range = range(df_PeMS.columns)
        columns = list(subgraph.nodes)
    else:
        dirpath = './experiments/clusterSubGraph'
        nodes_range = range(len(df_PeMS.columns)+1-cluster_size)
        columns = df_PeMS.columns[i:i+cluster_size]

    filename = Path(dirpath) / f"S{cluster_size}l{train_len}"
    
    if (filename.isfile()):

        
    
        cluster_dict={"size":cluster_size}

        for i in nodes_range:
            model = LSTMModel(cluster_size, 32 ,cluster_size, layers)
            train_loader, val_loader, test_loader = createLoaders(df_PeMS, columns,  perc_train, perc_val)
            cluster_dict[i]={"model":model,"train":train_loader,"val":val_loader,"test":test_loader}

        with open(filename, 'wb') as f:
            pickle.dump(cluster_dict, f)

    return model, train_loader, val_loader, test_loader


from torch.utils.data import Dataset
class TimeSeriesDataset(Dataset):
    
    
    def __init__(self, data, window_size, stride, prediction_horizon=1):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return (len(self.data) - self.window_size - self.prediction_horizon) // self.stride + 1

    def __getitem__(self, index):
        # Calculer le début et la fin de la fenêtre d'entrée
        start = index * self.stride
        end = start + self.window_size

        # Extraire les données d'entrée
        inputs = self.data[start:end]

        # Ajouter du padding ou du troncage si nécessaire pour avoir une taille fixe
        if len(inputs) < self.window_size:
            inputs = np.pad(inputs, (0, self.window_size - len(inputs)), 'constant')
        elif len(inputs) > self.window_size:
            inputs = inputs[:self.window_size]

        # Convertir les données d'entrée en tenseur PyTorch
        inputs = torch.from_numpy(inputs).float()

        # Calculer le début et la fin de la fenêtre de sortie
        start = end
        end = start + self.prediction_horizon

        # Extraire les données de sortie
        targets = self.data[start:end]

        # Ajouter du padding ou du troncage si nécessaire pour avoir une taille fixe
        if len(targets) < self.prediction_horizon:
            targets = np.pad(targets, (0, self.prediction_horizon - len(targets)), 'constant')
        elif len(targets) > self.prediction_horizon:
            targets = targets[:self.prediction_horizon]

        # Convertir les données de sortie en tenseur PyTorch
        targets = torch.from_numpy(targets).float()
        return inputs, targets

    
def my_data_loader(data, window_size = 7, stride = 1,prediction_horizon=1,batch_size=32):
    from torch.utils.data import DataLoader

    """
    Create a Time Serie DataLoader and format it correctly if CUDA is available GPU else CPU

    Parameters
    ----------
    data : pd.Dataframe
        dataframe with all the PeMS data
    windows_size : int
        Sliding window use for training
    stride : int
        the amount of movement after processing each sliding windows
    prediction_horizon : int 
        size of the target values of each sliding windows

    """
    
    dataset = TimeSeriesDataset(data.values, window_size, stride, prediction_horizon)
    loader = DataLoader(dataset, batch_size, shuffle=False)
    if torch.cuda.is_available():
        loader = [(inputs.to(device), targets.to(device)) for inputs, targets in loader]
    return loader

def createLoaders(df_PeMS, columns=0, perc_train = 0.7, perc_val = 0.15,  window_size = 7, stride = 1, prediction_horizon=1, batch_size=32):
    """
    Returns torch.DataLoader for train validation and test data
    
    Parameters
    ----------
    df_PeMs : pd.Dataframe
        dataframe with all the PeMS data
    columns : List 
        List of columns to process
    windows_size : int
        Sliding window use for training
    stride : int
        the amount of movement after processing each sliding windows
    prediction_horizon : int 
        size of the target values of each sliding windows
    """
    
    if not columns:
        columns = df_PeMS.columns
        
    train_len = len(df_PeMS)

    train_data= df_PeMS[columns][:int(train_len * perc_train)]
    val_data =  df_PeMS[columns][int(train_len * perc_train): int(train_len * (perc_train + perc_val))]
    test_data = df_PeMS[columns][int(train_len * (perc_train + perc_val)):]
    
    train_loader = my_data_loader(train_data, window_size, stride, prediction_horizon, batch_size)
    val_loader = my_data_loader(val_data, window_size, stride, prediction_horizon, batch_size)
    test_loader = my_data_loader(test_data, window_size, stride, prediction_horizon, batch_size)

    return train_loader, val_loader, test_loader, test_data



def load_PeMS04_flow_data(input_path: Path = "./data/PEMS04/"):
    import pandas as pd
    import numpy as np
    
    """
    
    Function to load traffic flow data from 'npz' and 'csv' files associated with PeMS

    Parameters
    ----------
    input_path: Path
        Path to the input directory

    Returns
    -------
    df_PeMS : pd.Dataframe
        With the flow between two sensors
    
    df_distance:
        Dataframe with the distance metrics between sensors
    
    """


    flow_file = input_path + 'pems04.npz'
    csv_file  = input_path + 'distance.csv'

    # the flow data is stored in 'data' third dimension
    df_flow = np.load(flow_file)['data'][:,:,0]
    df_distance = pd.read_csv(csv_file)
    
    dict_flow = { k : df_flow[:,k] for k in range(df_flow.shape[1])}

    df_PeMS = pd.DataFrame(dict_flow)


    start_date = "2018-01-01 00:00:00"
    end_date = "2018-02-28 23:55:00"
    interval = "5min"
    index = pd.date_range(start=start_date, end=end_date, freq=interval)
    df_PeMS = df_PeMS.set_index(index)

    return df_PeMS, df_distance


def local_dataset(df, nodes=[], perc_train = 0.7, perc_val = 0.15,  window_size = 7, stride = 1, prediction_horizon=1, batch_size=32):
    """
    Create datasets and data loaders for training, validation, and test sets

    Parameters
    ---------
    df : pd.Dataframe
        A Dataframe with the time series for all the nodes
    nodes : list
        A list of columns ids to process
    """

    import pandas as pd
    import warnings

    if len(nodes) == 0 or not (set(nodes).issubset(set(df.columns))):
        warnings.warn("Nodes selected not in dataset or empty filter, processing all nodes")
        nodes = df.columns
 
    data_dict={}
    counter = 0 

    for i in nodes: 
        
        train, val, test, test_data = createLoaders(pd.DataFrame(df.loc[:,i]),
                                         perc_train = perc_train,
                                         perc_val = perc_val,
                                         window_size = window_size,
                                         stride = stride, 
                                         prediction_horizon = prediction_horizon,
                                         batch_size = batch_size )
        
        
        data_dict[counter]={'train':train,'val':val,'test':test, "test_data" :test_data }
        counter = counter + 1

    return data_dict


def preprocess_PeMS_data(df_PeMS, time_serie_percentage_length, df_distance, init_node : int = 0, n_neighbors : int = 99,
                        smooth = True, center_and_reduce = False, normalize = False, sort_by_mean = True):
    from src.utils_graph import create_graph, subgraph_dijkstra, compute_adjacency_matrix

    """
    Filter to n nearest neightbors from 'init_node', sort by mean traffic flow, and normalize and smooth data

    Parameters
    ----------
    time_serie_percentage_length: float
        Percentage of the time series we want to keep for training


    df_distance: pandas.DataFrame
        Dataframe with the distance between nodes
    
    init_node : int
        Index of the node we want to start with

    n_neighbors: int
        Number of nearest neighbors to consider

    smooth: boolean
        Flag for smoothing the time series

    center_and_reduce: boolean
        Flag for centering and reducing the data

    normalize: boolean
        Flag for normalizing the data

    sort_by_mean: boolean
        Flag for sorting the data by mean

    Returns
    ----------
    df_PeMS :
        PeMS that have been preprocessed

    adjacency_matrix : array
        the adjacency matrix of PeMS

    meanstd_dict : dictionary
        Dictionary containing the mean and std of the prenormalize DataFrame
    """

    df_PeMS = df_PeMS[:int(len(df_PeMS)* time_serie_percentage_length)]

    # Filter nodes to retain only n nearest neighbors
    graph = create_graph(df_distance)
    if n_neighbors :
        graph = subgraph_dijkstra(graph, init_node, n_neighbors)
        df_PeMS = df_PeMS[list(graph.nodes)]

    #Sort data by mean traffic flow
    if sort_by_mean:   
        df_sorted= df_PeMS.mean().sort_values()
        index_mean_flow = df_sorted.index
        column_order = list(index_mean_flow)
        df_PeMS = df_PeMS.reindex(columns = column_order)
    
    adjacency_matrix = compute_adjacency_matrix(graph,list(df_PeMS.columns))

    if smooth :
        df_PeMS = exp_smooth(df_PeMS)
    
    if center_and_reduce :
        df_PeMS, meanstd_dict = center_reduce(df_PeMS)
        return df_PeMS, adjacency_matrix, meanstd_dict
    elif normalize :
        df_PeMS = normalize_data(df_PeMS)
        
    return df_PeMS, adjacency_matrix

def plot_prediction(y_true, y_pred, test_data,meanstd_dict, window_size, time_point_t=0,  node = 0, plot_fig_name = 'plot.jpg'):

    """
    Simple function for a line plot of actual versus prediction values

    Parameters
    ---------- 
    y_true, y_pred : array
        true value and predicted value to plot. The array are in shape 
        (length of time serie, horizon of prediction , time serie dimension)
    meanstd_dict : dictionary
        Dictionary containing the mean and std of the test data to unnormalize  
    window_size : 
        size of the sliding window
    time_point_t : int 
        Which time point to start ploting
    node : int
        node number/time serie dimension to plot in case of a multivariate forecasting

    """

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    index = test_data.index
    test_data = test_data* meanstd_dict['std'] + meanstd_dict['mean']
    prediction_horizon = y_true.shape[1]
    if y_true.shape[1:] == (1,1) :
            plt.figure(figsize=(30, 5))
            plt.title(f'Actual vs Prediction ')
            plt.plot(y_true[:,0,0],label='Actuals')
            plt.plot(y_pred[:,0,0], label='Predictions')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            
    else:
        window_start = time_point_t
        window_end = time_point_t + window_size
        plt.figure(figsize=(20, 9))
        plt.title(f' Actual vs Prediction ')
        # plot y_true as scatter plot with lines
        plt.scatter(index[window_end:window_end+prediction_horizon],y_true[time_point_t,:,node],color='green',label='Actuals')
        plt.plot(index[window_end:window_end+prediction_horizon],y_true[time_point_t,:,node],color='green', linestyle='-', linewidth=1)
        # plot y_pred as scatter plot with lines
        plt.scatter(index[window_end:window_end+prediction_horizon],y_pred[time_point_t,:,node],color='red', label='Predictions')
        plt.plot(index[window_end:window_end+prediction_horizon],y_pred[time_point_t,:,node], color='red', linestyle='-', linewidth=1)

        # plot a grey area for a sliding window
        plt.plot(index[window_start:window_end+1], test_data[window_start:window_end+1], label='y_true')
        plt.axvspan(index[window_start], index[window_end], alpha=0.1, color='gray')
        plt.plot()
        ax = plt.gca()
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        # Specify the file name and format
    # Set the properties of the x-axis
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Customize the x-axis tick labels
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('Temps (5 minutes intervales)')
    plt.ylabel('Traffic Flow')
    plt.title("{} : Prediction for the {}".format(plot_fig_name.split('.')[0], index[window_end].strftime('%Y-%m-%d')), fontsize=18, fontweight='bold')
    plt.savefig(plot_fig_name)  
    # Close the plot to free up memory (optional)
    plt.close()


def unormalize_center_reduce(y_pred, y_true, meanstd_dict, sensor_order_list):
    if len(sensor_order_list) > 1:
        for k in range(len(sensor_order_list)):
            y_pred[k] = y_pred[k] * meanstd_dict[sensor_order_list[k]]['std'] + meanstd_dict[sensor_order_list[k]]['mean']
            y_true[k] = y_true[k] * meanstd_dict[sensor_order_list[k]]['std'] + meanstd_dict[sensor_order_list[k]]['mean']
    elif len(sensor_order_list) == 1:
        y_pred = y_pred * meanstd_dict[sensor_order_list[0]]['std'] + meanstd_dict[sensor_order_list[0]]['mean']
        y_true = y_true * meanstd_dict[sensor_order_list[0]]['std'] + meanstd_dict[sensor_order_list[0]]['mean']
    return y_true, y_pred


def normalize_center_reduce(y_pred, y_true, meanstd_dict, sensor_order_list):
    if len(sensor_order_list) > 1:
        for k in range(len(sensor_order_list)):
            y_pred[k] = (y_pred[k] - meanstd_dict[sensor_order_list[k]]['mean']) / meanstd_dict[sensor_order_list[k]]['std']
            y_true[k] = (y_true[k] - meanstd_dict[sensor_order_list[k]]['mean']) / meanstd_dict[sensor_order_list[k]]['std']
    elif len(sensor_order_list) == 1:
        y_pred = (y_pred - meanstd_dict[sensor_order_list[0]]['mean']) / meanstd_dict[sensor_order_list[0]]['std']
        y_true = (y_true - meanstd_dict[sensor_order_list[0]]['mean']) / meanstd_dict[sensor_order_list[0]]['std']
    return y_true, y_pred
