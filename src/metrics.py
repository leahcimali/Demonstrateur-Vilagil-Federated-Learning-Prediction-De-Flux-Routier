import numpy as np
from sklearn.metrics import  mean_squared_error


def rmse(y_true,y_pred):
    """
    Root mean square error calculate between y_pred and y_true
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmspe(y_true, y_pred, EPSILON=0):
    # The epsilon parameter move the time series away from zero values of values epsilon
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + EPSILON)))) * 100)

def mape(y_true, y_pred, EPSILON=0):
    # The epsilon parameter move the time series away from zero values of values epsilon
    return np.mean(np.abs((y_true - y_pred)/(y_true + EPSILON)))*100   

def maape(y_true,y_pred,EPSILON=0):
    # The epsilon parameter move the time series away from zero values of values epsilon
    return (np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true +EPSILON))))*100)

def rmsse(y_true, y_pred):
    # Calculate the numerator (RMSE)
    numerator = np.sqrt(np.mean(np.square(y_true - y_pred)))

    # Calculate the denominator (scaled error)
    denominator = np.sqrt(np.mean(np.square(y_true[1:] - y_true[:-1])))

    # Calculate the RMSSE
    rmsse = numerator / denominator

    return rmsse



def calculate_metrics(y_true, y_pred,percentage_error_fix =0):
    from src.metrics import rmse, rmspe, maape, mape 
    from sklearn.metrics import mean_absolute_error
    
    """
    Parameters: 
    -----------

    Parameters
    ----------

    y_true : float
        True value

    y_pred : float
        Predicted value

    percentage_error_fix : float
        Add a float to the time serie for calculation of percentage because of null values 

    Returns
    -------
    Dict :
        A dictionary with rmse, rmspe, mae, mape and maape values
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    metric_dict={}
    rmse_val= rmse(y_true, y_pred)
    rmspe_val = rmspe(y_true,y_pred,percentage_error_fix)
    mae_val = mean_absolute_error(y_true,y_pred)
    mape_val = mape(y_true,y_pred,percentage_error_fix)
    maape_val =  maape(y_true,y_pred,percentage_error_fix)
    
    metric_dict = {"RMSE":rmse_val, "RMSPE": rmspe_val, "MAE":mae_val,"MAPE":mape_val, "MAAPE": maape_val}
    return metric_dict
