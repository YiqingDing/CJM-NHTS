import func, utils, xlsxwriter, pandas, pathlib, os
###################### Test Packages #######################
###################### Input #######################
#Input for the clustering algorithm:
	# mc_ls: List of lists, each entry is a discrete time Markov chain 
	# s: Cardinality of state space
	# alpha: Global precision
#################### Matlab Data Input ##################
# raw_trip_file = 'SimulatedData/matlab_data.csv' #File name of the matlab file
# mc_ls = utils.csv_read(raw_trip_file, output_type = tuple)
# s = 5
# alpha = 80 #Global precision (this val equals to 1/s for alpha_kij)
# mc_crop_ls = [mc_ls]
# print('MC crop list generated!')
#################### NHTS Data Input ####################
t_interval = 0.5
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
complete_trip_file = 'trippub.csv' #File name of the complete data
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset
trip_df = func.tripls2df(trip_ls_raw, t_interval) #Convert trips into df where col are time windows
s = 21
alpha = 10 #Global precision (this val equals to 1/s for alpha_kij)
# loop_iter = range(47) #Max transition number available
# Use user input for min and max loop number
loop_min = int(input('Please enter the min loop number(inclusive): ') or 0)
loop_max = int(input('Please enter the max loop number(exclusive - max 47): ') or 47)
loop_iter = range(loop_min,loop_max)
##### Complete Dataset for Prior Generation #####
#### The following 4 lines generate the complete data file and save it as a csv (commented)
# raw_trip_file_complete = 'trippub.csv' #File name of the 2k data
# trip_ls_raw_complete  = func.trip_ls_input(raw_trip_file_complete,'w',save_file = False) #Generate the day trips for the complete dataset
# trip_df = func.tripls2df(trip_ls_raw_complete, t_interval)
# trip_df.to_csv('trip_df_complete.csv', index = False)
#### The above 4 lines generate the complete data file and save it as a csv (commented)
trip_df_complete = pandas.read_csv('trip_df_complete.csv').iloc[trip_df.shape[0]:,] #Only use the rows not belong to test dataset
trip_df_select = trip_df_complete.sample(3000) #Choose 3000 samples (select the size of prior dataset - how much prior info given)
trip_df_prior = trip_df_complete #Use trip_df_complete or trip_df_select
#################################################
# Write to an excel in Parent/Results/Bayesian/Bayesian_Clustering_Results.xlsx
workbook = xlsxwriter.Workbook(str(pathlib.Path(os.getcwd()).parent)+'/Results/Bayesian/Bayesian_Clustering_Results_'+os.environ.get('USER')+'.xlsx')
worksheet_0 = workbook.add_worksheet('General Results')
worksheet_0.write_row(0,0,['','hello world'])
workbook.close()