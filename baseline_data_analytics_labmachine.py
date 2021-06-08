import func, os, pathlib, importlib

# Input data file (raw trip list) - use this for plot
raw_file_name = 'trippub_top2k.csv' #Raw file name of the 2k data
raw_trip_ls  = func.trip_ls_input(raw_file_name, 'r') #Generate the day trips for dataset

# Input: Local data files from "output" folder with names similar as "FinalResult0.csv"
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
# input_data_folder = '2020-11-10' #Data files are located in folders named after this var
input_data_folder = 'Compiled' #Data files are located in folders named after this var
# input_data_folder = 'Test' #Data files are located in folders named after this var
sorted_file_name = 'result_simple_sorted.csv'

input_folder_path = str(current_path.parent)+ '/LabMachineResults/'+input_data_folder #Data file path (folder name)
sorted_file_path = input_folder_path+'/'+sorted_file_name # Sorted file output path
sorted_result = func.data_sort_labmachine(folder_path = input_folder_path, data_name_format = 'FinalResult', output_file_name= sorted_file_name) #compute the sorted result and output a dataframe (results may have been converted to str)

# Plot sorted_result and save the plots to certain folder
func.plot_vec_centers(trip_data_df = sorted_result, result_loc = input_folder_path, raw_trip_ls = raw_trip_ls, top_n = 5)

# Result translation
simplified_translation = False
result_translated = func.data_translate_labmachine(sorted_file_path, simplified_translation, False) #Translate data into words 
if not simplified_translation: #If translated in full words
	sort_dict, sort_ls = func.most_frequent_activities(result_translated, sorted_file_path)
	func.plot_freq_centers_bar(sort_dict, result_loc = input_folder_path)
	func.plot_freq_centers_heat(sort_dict, result_loc = input_folder_path)
	# print(sort_ls)
	