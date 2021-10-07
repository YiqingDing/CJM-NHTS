import func, os, pathlib, importlib
# This is the analysis file for baseline approach's EXCEL results.
# For analysis using folders, please refer to baseline_data_analytics_labmachine.py in Archive.

# Input: Read result files from Result folder 
dataFileNameList = []
dataFileNameList.append('2021-09-06 iris.xlsx')
dataFileNameList.append('2021-09-06 irislab.xlsx')
dataFileNameList.append('2021-09-07 iris.xlsx')
dataFileNameList.append('2021-09-07 irislab.xlsx')

input_folder_path = str(pathlib.Path(os.getcwd()).parent)+'/Results/Baseline_LabMachine/' #Result folder path

# input_data_folder = 'Test' #Data files are located in folders named after this var
sorted_file_name = 'result_simple_sorted.csv'
FinalT = pd.DataFrame(columns = ['Trial Number','Key Identifier','Best CJM in Trial','Score of Best CJM'])

for dataFileName in dataFileNameList:
	dataFilePath = input_folder_path+ dataFileName # Sorted file output path
	DataT = pd.read_excel(dataFilePath, sheet_name = 0, header = 1)
	FinalT = pd.concat([FinalT,DataT])

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