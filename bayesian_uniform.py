import func, utils, xlsxwriter, pandas, pathlib, os, time
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
loop_min = int(input('Please enter the min number of transitions(inclusive - min 1): ') or 1)
loop_max = int(input('Please enter the max number of transitions(inclusive - max 47): ') or 47)
loop_iter = range(loop_min,loop_max+1)
suffix = input('Please enter any suffix for the output file name: ')
# suffix = '_'+suffix if suffix else '' #Add underscore if suffix is nonempty
##### Complete Dataset for Prior Generation #####
#### The following 4 lines generate the complete data file and save it as a csv (commented)
# raw_trip_file_complete = 'trippub.csv' #File name of the 2k data
# trip_ls_raw_complete  = func.trip_ls_input(raw_trip_file_complete,'w',save_file = False) #Generate the day trips for the complete dataset
# trip_df = func.tripls2df(trip_ls_raw_complete, t_interval)
# trip_df.to_csv('trip_df_complete.csv', index = False)
#### The above 4 lines generate the complete data file and save it as a csv (commented)
# trip_df_complete = pandas.read_csv('trip_df_complete.csv').iloc[trip_df.shape[0]:,] #Only use the rows not belong to test dataset
# sample_size = int(input('Please enter number of samples to be selected from complete dataset as prior (default all):') or 0)
# if sample_size == 0:
	# trip_df_prior = trip_df_complete #Use trip_df_complete or trip_df_select
# else:	 
	# trip_df_prior = trip_df_complete.sample(sample_size) #Choose samples with input sample_size (select the size of prior dataset - how much prior info given)
#################################################
# File names and paths
workbook_path =str(pathlib.Path(os.getcwd()).parent)+'_'.join(['/Results/Bayesian/Bayesian_Clustering_Results',os.environ.get('USER'),suffix])+'.xlsx'
raw_result_path = 'output/raw/' #File path to save raw result
id_dict_path = 'output/idDict/' #File path to save id_dict (in Bayesian clustering)
# Removes all existing output files to avoid conflicts
shutil.rmtree(raw_result_path)
shutil.rmtree(id_dict_path)
#################################################
# Write to an excel in Parent/Results/Bayesian/Bayesian_Clustering_Results.xlsx
workbook = xlsxwriter.Workbook(workbook_path)
worksheet_0 = workbook.add_worksheet('General Results')
print('Uniform execution starts! Transition number from',loop_min,'to',loop_max)
print('***********************************************************************************************')
for i, mc_len in enumerate(loop_iter): #Iterate over different number of transitions
	last_time = time.time()
	# mc_len = 4 #Test mc_len value
	mc_crop_dict, mc_title_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to a dict of mc lists using number of transitions (keyed by window index), mc_title_ls is list of titles, index based on order of mc_crop_dict's values
	# mc_crop_dict_prior = func.tripdf2mcls(trip_df_prior, mc_len)[0] #Convert the complete trip df to dict of mc lists (keyed by window index)
	print('MC crop list generated for mc_len=',mc_len,'!')
	worksheet_1 = workbook.add_worksheet(str(mc_len)+' Transition') #Added new sheet to record result for the specific number of transitions
	last_row_no = 1 # Last row number used for writing in worksheet_1 (reset for every mc_len)
	###################### Input #######################
	set_no = len(mc_crop_dict.keys()) #Total number of cluster sets for one day (# of time windows)
	cluster_len_ls = [] #List of cluster length
	for idx, (window_idx, mc_ls) in enumerate(mc_crop_dict.items()): #Iterate over all the time windows (idx for titles, window_idx for window index)
		# Each time window contains a list of MCs
		print('--------------Clustering Starts for No.'+str(idx+1)+' out of '+str(set_no) +' sets--for MC of length '+str(mc_len)+'---------')
		# mc_ls_prior = mc_crop_dict_prior[window_idx] #Get the prior mc list (can be empty list)
		# prior_input_dev = ['dev', mc_ls_prior] if mc_ls_prior else ['uniform'] #Generate the prior input for dev prior (if no prior exists, use uniform)
		# Perform Bayesian clustering (prior using the dataset )
		# cluster_ls, trans_ls = func.bayesian_clustering(mc_ls,alpha, s, prior_input = prior_input_dev, KL_dict = {'suffix':suffix})
		
		clustering_result = func.bayesian_clustering(mc_ls,alpha, s, KL_dict = {'id_dict_path':id_dict_path, 'id_suffix': str(mc_len)+'_'+str(idx+1)}) #Uniform prior
		cluster_ls = clustering_result['cluster_ls']
		trans_ls = clustering_result['trans_ls']
		cluster_len_ls.append(len(cluster_ls)) #Append cluster length
		print('The number of clusters is',cluster_len_ls[idx])
		
		# Saving to worksheet starts from row 1 (1st row reserved for general result)
		# row_0 = 1+ idx * (s+1) #Starting row number of current saving 
		current_title = ['No. '+str(idx+1)] + mc_title_ls[idx] #Current title includes a number and title time windows
		worksheet_1.write_row(last_row_no,0,current_title) #Write current title (time window), at row 1, 23, etc.
		if cluster_len_ls[-1]>1: #Only saves trans_ls if the clustering result is meaningful
			utils.dict2json(raw_result_path + '_'.join(['bayesian_raw_results',suffix, str(mc_len),str(idx+1)]) + '.json', clustering_result['cluster_ls_id']) #Save clustering result (in id format) to output/raw/
			trans_ls_zip = list(zip(*trans_ls)) #Convert flat trans_ls to a list in which each entry is a list of row values for all trans matrices
			k0 = 0 #Index for row no of trans_ls_zip - relative row number (reset for each time window)
			for j in range(last_row_no+1,last_row_no+s+1): #Absolute row number in excel
				# j starts from last_row_no+1 (2, 24, etc.) and ends at last_row_no+s+1 (22, 44, etc.)
				# k0 points to each row within trans_ls_zip and k0 increments with j
				for k, row_ls in enumerate(trans_ls_zip[k0]): #Iterate over list of rows for all trans_mat
					worksheet_1.write_row(j,k*(s+1),row_ls) #Write the one row for one matrix and goes to the next matrix
				k0 +=1 #Update the relative index
			last_row_no += s+1 #Update the last_row_no with the new index
		else:
			worksheet_1.write_row(last_row_no+1,0,['','No meaning clustering result generated!'])
			last_row_no += 2
	idx_meaningful = [i+1 for i, e in enumerate(cluster_len_ls) if e != 1]
	print('The number of clusters for this division are',cluster_len_ls,'and the index of meaningful clusters are',idx_meaningful)
	# Saves the file
	worksheet_1.write_row(0,0,[str(mc_len)+' transitions:', str(cluster_len_ls), str(idx_meaningful)]) #Write on the time window specific sheet
	worksheet_0.write_row(i,0,[str(mc_len)+' transitions:', str(cluster_len_ls), str(idx_meaningful)]) #Write on the 'General Result' sheet
	
	print('Time spent on this dataFile is',time.time() - last_time)
	last_time = time.time()
###################### Test #######################
print('***********************************************************************************************')
print('Uniform execution completed! Transition number from',loop_min,'to',loop_max)

workbook.close()