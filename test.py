import func, os, pathlib, utils, ast
import matplotlib.pyplot as plt

# Input data file (raw trip list) - use this for plot
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
data_file = 'trippub_top2k.csv' #File name of the 2k data
file_trip = str(current_path.parent.parent)+'/Data/'+data_file #Data file location+name
trip_ls  = func.data_processing(file_trip) #Generate the day trips for dataset

# Input: Local data files from "output" folder with names similar as "FinalResult0.csv"
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
# input_data_folder = '2020-11-10' #Data files are located in folders named after this var
input_data_folder = 'Compiled' #Data files are located in folders named after this var
# input_data_folder = 'Test' #Data files are located in folders named after this var
sorted_file_name = 'result_simple_sorted.csv'

input_folder_path = str(current_path.parent)+ '/LabMachineResults/'+input_data_folder #Data file path (folder name)
sorted_file_path = input_folder_path+'/'+sorted_file_name

simplified_translation = False


