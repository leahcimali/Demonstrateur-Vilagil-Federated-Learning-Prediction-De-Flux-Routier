"""
Use to download the dataset.
"""

import os


import pandas as pd
import json
import requests
import csv

os.makedirs("./dataset", exist_ok=True)
# Contains the connection between the id and the name of a sensor display on the map of AVATAR Cerema website
sensors = pd.read_csv("../data/count_points.csv", sep=";")

# Contains a lot of informations but the more important are the name of the sensors
with open('./data/Rennes_sensors_graph.json') as json_file:
    sensors_graph = json.load(json_file)

id_sensors = []
for sensor in sensors_graph["nodes"]:
    id_sensors.append(sensors[sensors["count_point_name"] == sensor["key"]]["count_point_id"].values[0])

for id_s in id_sensors:
    id_sensor = id_s  # id of the sensors != name of the sensor
    start_time = "2017-01-01"
    end_time = "2023-07-17"
    csv_file_path = f'./dataset/{id_sensor}_measures.csv'

    path_api = f"https://avatar.cerema.fr/api/fixed_measures/download?as_vehicle_nb=true&end_time={end_time}&start_time={start_time}&count_point_ids={id_sensor}"

    r = requests.get(path_api)
    decoded_content = r.content.decode('utf-8')

    # Save the decoded_content as a CSV file
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in decoded_content.splitlines():
            writer.writerow(line.split(','))

    print("CSV file saved successfully.")
