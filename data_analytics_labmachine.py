import func, os, pathlib

# Input: Local data files from "output" folder with names similar as "FinalResult0.csv"
current_path = pathlib.Path(os.getcwd()) #Get the current working directory
input_data_date = '2020-11-10' #Data files are located in folders named after its running date
sorted_file_name = 'result_sorted.csv'

folder_path = str(current_path.parent)+ '/LabMachineResults/'+input_data_date #Data file path (folder name)
sorted_file_path = folder_path+'/'+sorted_file_name

sorted_result = func.data_sort_labmachine(folder_path = folder_path, data_name_format = 'FinalResult', file_name= sorted_file_name)
result_translated=func.data_translate_labmachine(sorted_file_path, True, False)

# Visualize the cluster data


# Visualize the raw data
# Extract raw data

#Plot
# test_lines = (((1,1),(2,2)), ((1,2),(2,2)))
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# for trip_ind in trip_ls:
# 	plt.plot(trip_ind[0],trip_ind[1])
# plt.show()
