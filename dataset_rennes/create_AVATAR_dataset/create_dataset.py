import numpy as np
import pandas as pd
from glob import glob


measures = {}
for path in glob('../dataset/*'):
    id_sensor = pd.read_csv(path, sep=";").head(1)["count_point_id"].item()
    measures[id_sensor] = pd.read_csv(path, sep=";", usecols=["measure_datetime", "veh_nb"])

for sensor in measures:
    measures[sensor]["measure_datetime"] = pd.to_datetime(measures[sensor]["measure_datetime"], format='%Y-%m-%dT%H:%M:%S%z')
    measures[sensor].index = measures[sensor]["measure_datetime"]
    measures[sensor].drop(columns=["measure_datetime"], inplace=True)
    measures[sensor] = measures[sensor].resample('5T').interpolate(method="polynomial", order=5)
    measures[sensor].rename(columns={"veh_nb": f"{sensor}"}, inplace=True)

final_df = measures[97]
for sensor in measures:
    if sensor != 97:
        final_df = final_df.merge(measures[sensor], left_index=True, right_index=True, how='inner')

final_df = final_df.to_numpy()
np.savez('dataset_Rennes.npz', data=final_df)