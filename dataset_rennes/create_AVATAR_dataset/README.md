# How to create the Rennes dataset

### First
```py
python download_sensors_measures.py
```
this download all the CSV in a folder dataset/

### Second
```py
python create_dataset.py
```
this will resample all the data to put every time series on the same time scale to finaly create a npz file with all time series in it.


### Or just dowload it here
https://drive.google.com/drive/folders/1LZhNpsbG5m1RjlNA7nW6Wof6MWM3TM9I