# -*- coding: utf-8 -*-
"""120554-104966 - Sandya Vaigundam Sadasivam - Apr 25, 2023 1135 PM - Big Data - SPN Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
	https://colab.research.google.com/drive/1WO9SaOBK5uvEvOq4tJYUdWb3GxdRlIiT
"""

# Main File for Learned Index
import numpy as np
import pandas as pd
import time
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.LearningWrappers import learn_parametric
from spn.algorithms.Statistics import get_structure_stats
from spn.io.Graphics import plot_spn
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
import psutil
import random  # for self test
from inputFileProcess import *

print("Input Dataset: D1/D2/D3 ?")
datasetName = input()
dataframe = pd.read_csv("./dataset_csv/int_10000_20230717_123124_train_aifeynman.ssv", header=None)
column_labels = dataframe.keys()
column_labels = ["x", "y"]
dataframe = dataframe.set_axis(column_labels, axis=1)


# Switch x and y columns, i.e. 0th and 1st column, @author: Avin
col_list = list(dataframe)
col_list[0], col_list[1] = col_list[1], col_list[0]
dataframe.columns = col_list
# print(dataframe.head)

train_data = dataframe.to_numpy()
# test_data=train_data[:1000]
num_vars = train_data.shape[1]  # Number of variables in the dataset
ds_context = Context(parametric_types=[Gaussian, Gaussian]).add_domains(train_data)

start_time=time.time()
spn = learn_parametric(train_data, ds_context)
end_time=time.time()
total_build_time = round((end_time-start_time) * 1000, 2) #in ms
print("Build Time:", total_build_time, " ms")

print("Structure Stats:",get_structure_stats(spn))
# print("SPN Structure")
# plot_spn(spn, 'basicspn.png')

column_data_as_list = train_data
column_data_as_list = column_data_as_list.tolist()
noOfQueries = 300
query_point = random.sample(column_data_as_list,  noOfQueries)

start_time=time.time()
ll = log_likelihood(spn, np.asarray(query_point))
end_time=time.time()
total_query_time = round((end_time-start_time) * 1000, 2) #in ms
print("time taken to search for", len(query_point), " Keys" , f"{total_query_time: .2f}", " ms")
avg_query_exec_time = total_query_time/noOfQueries
print("Avg query execution time: ", total_query_time, " ms")
# print(ll)

# Get the current process object
process = psutil.Process()
# Get the memory usage in bytes
mem_info = process.memory_info().rss
# Convert to megabytes
mem_usage = mem_info / 1024 / 1024
print(f"Memory usage: {mem_usage:.2f} MB")

jsonObj = read_json_file()
jsonObj = modify_json_data(read_json_file(), 'SPN', datasetName, total_build_time, total_query_time)
write_json_file(jsonObj)