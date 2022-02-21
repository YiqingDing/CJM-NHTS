import func, utils, openpyxl, pandas, pathlib, os, sys, shutil, time
import numpy as np
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
# Use user input for min and max loop number
# loop_iter = range(47) #Max transition number available
loop_start = int(input('Please enter the starting number of transitions(inclusive - min 1): ') or 1)
loop_end = int(input('Please enter the ending number of transitions(inclusive - max 47): ') or 47)
sample_size = int(input('Please enter number of samples to be selected from complete dataset as prior (default or 0 uniform, -1 for complete dataset):') or 0)
suffix = input('Please enter any suffix for the output file name: ')
col_empty = np.full((s,1),None) #Create a column of empty cells for filling
########## Complete Dataset for Prior Generation ##########
#### The following 4 lines generate the complete data file and save it as a csv (commented)
# raw_trip_file_complete = 'trippub.csv' #File name of the 2k data
# trip_ls_raw_complete  = func.trip_ls_input(raw_trip_file_complete,'w',save_file = False) #Generate the day trips for the complete dataset
# trip_df = func.tripls2df(trip_ls_raw_complete, t_interval)
# trip_df.to_csv('trip_df_complete.csv', index = False)
#### The above 4 lines generate the complete data file and save it as a csv (commented)
# File names and paths
workbook_path =str(pathlib.Path(os.getcwd()).parent)+'_'.join(['/Results/Bayesian/Bayesian_Clustering_Results',os.environ.get('USER'),suffix])+'.xlsx'
folder_path = os.path.split(workbook_path)[0] #Extract the folder path for the workbook
pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True) #Create the folder (and parent folder) if not exists yet 
if os.path.exists(workbook_path): #Remove workbook if it already exists
	os.remove(workbook_path)
raw_result_path = str(pathlib.Path(os.getcwd()).parent) + '/Results/Bayesian/raw/raw/' #File path to save raw result
id_dict_path = str(pathlib.Path(os.getcwd()).parent) + '/Results/Bayesian/raw/idDict/' #File path to save id_dict (in Bayesian clustering)
prior_path = str(pathlib.Path(os.getcwd()).parent) + '/Results/Bayesian/prior/' #File path to save priors
pathlib.Path(prior_path).mkdir(parents=True, exist_ok=True) #Create the folder (and parent folder) if not exists yet 
pathlib.Path(raw_result_path).mkdir(parents=True, exist_ok=True) #Create the folder (and parent folder) if not exists yet 
pathlib.Path(id_dict_path).mkdir(parents=True, exist_ok=True) #Create the folder (and parent folder) if not exists yet 
# Removes all existing output files to avoid conflicts
# shutil.rmtree(raw_result_path,ignore_errors=True)
# shutil.rmtree(id_dict_path,ignore_errors=True)
#################################################
trip_df_complete = pandas.read_csv('trip_df_complete.csv').iloc[trip_df.shape[0]:,] #Only use the rows not belong to test dataset
loop_iter = range(loop_start,loop_end+1) if loop_end >= loop_start else range(loop_start,loop_end-1, -1) #The iterator always starts from loop_start and ends with loop_end (both inclusive)
# suffix = '_'+raw_suffix if raw_suffix else '' #Add underscore if suffix is nonempty
if sample_size == 0:
	sample_size = 'Uniform'
	trip_df_prior = pandas.DataFrame() #For uniform prior, we will use a empty df for prior trip_df
elif sample_size == -1:
	sample_size = 'Complete'
	trip_df_prior = trip_df_complete #Use trip_df_complete or trip_df_select
else: #Also save/read prior if we are using random samples
	prior_file_path = prior_path + 'prior_' + str(sample_size) + '.json' #File path for current prior
	# Read existing prior or generate and save new prior
	if pathlib.Path(prior_file_path).is_file(): #if a prior file already exists
		trip_df_prior = pandas.read_json(prior_file_path) #Read the json file
		print('Reading An Existing Prior File!!!')
	else: #if no such prior exists
		trip_df_prior = trip_df_complete.sample(sample_size) #Choose samples with input sample_size (select the size of prior dataset - how much prior info given)
		trip_df_prior.to_json(prior_file_path) #Save the prior file
#################################################
# Write to an excel in Parent/Results/Bayesian/Bayesian_Clustering_Results.xlsx
workbook = openpyxl.Workbook() #Create a workbook with openpyxl
worksheet_0 = workbook.active #Get the first worksheet
worksheet_0.title = 'General Results' #First worksheet to save overall results
worksheet_0.append(['Number of Transitions', 'Number of Meaningful Clusters for Each Time Window', 'Index of Meaningful Time Window']) #Append a header row
workbook.save(workbook_path) #First save the workbook before any results
print('Execution starts! Prior size =',sample_size,'transition number from',loop_start,'to',loop_end)
print('***********************************************************************************************')
start_time = time.time()
for mc_len in loop_iter: #Iterate over different number of transitions
	last_time = time.time()
	# mc_len = 4 #Test mc_len value
	mc_crop_dict, mc_title_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to a dict of mc lists using number of transitions (keyed by window index), mc_title_ls is list of titles, index based on order of mc_crop_dict's values
	mc_crop_dict_prior = False if trip_df_prior.empty else func.tripdf2mcls(trip_df_prior, mc_len)[0] #Convert the prior trip df to dict of mc lists (keyed by window index) (if trip_df_prior is not empty)
	print('MC crop list generated for mc_len = ',mc_len,'!')
	# Load workbook at the beginning of each loop (add a sheet in each loop)
	workbook = openpyxl.load_workbook(workbook_path) #Realod workbook
	worksheet_0 = workbook.active #Retrieve the first worksheet
	worksheet_1 = workbook.create_sheet(str(mc_len)+' Transition') #Added new sheet to record result for the specific number of transitions
	worksheet_1.append([str(mc_len)+' transitions:'])
	###################### Input #######################
	set_no = len(mc_crop_dict.keys()) #Total number of cluster sets for one day (# of time windows)
	cluster_len_ls = [] #List of number of clusters for time windows
	for idx, (window_idx, mc_ls) in enumerate(mc_crop_dict.items()): #Iterate over all the time windows (idx for titles because all the titles are used, window_idx for window index)
		prior_input_dev = ['dev', mc_crop_dict_prior[window_idx]] if mc_crop_dict_prior else ['uniform'] #Generate the prior input for dev prior (if no prior exists, use uniform)
		# Each time window contains a list of MCs
		print('--------------Clustering Starts for No.'+str(idx+1)+' out of '+str(set_no) +' sets/time windows--for MC of length '+str(mc_len)+'---------')
		# Perform Bayesian clustering (prior using the dataset)
		clustering_result = func.bayesian_clustering(mc_ls,alpha, s, prior_input = prior_input_dev, KL_dict = {'id_dict_path':id_dict_path, 'id_suffix': str(mc_len)+'_'+str(idx+1)})
		# clustering_result = func.bayesian_clustering(mc_ls,alpha, s, KL_dict = {'suffix':suffix}) #Uniform prior
		cluster_len_ls.append(len(clustering_result['cluster_ls'])) #Append current number of clusters to list
		print('The number of clusters is',cluster_len_ls[idx])
		# Saving to worksheet starts from row 1 (1st row reserved for general result)
		# row_0 = 1+ idx * (s+1) #Starting row number of current saving 
		cluster_size_ls = ['Total number of datapoints',str(len(mc_ls)),'Size of clusters: ',str([len(cluster) for cluster in clustering_result['cluster_ls']])] if cluster_len_ls[-1]>1 else [] #Number of datapoints in each cluster if the clustering result is meaningful
		posterior_str = ['Posterior (initial, 1-cluster, final)',str(clustering_result['posterior'])] #Posterior string
		current_title = ['No. '+str(idx+1)] + mc_title_ls[idx] + cluster_size_ls + posterior_str #Current title includes [number index, time windows title string, Total number of datapoints, Size of clusters (list), Posterior (initial, 1-cluster, final)] 
		worksheet_1.append(current_title) #Append current title
		if cluster_len_ls[-1]>1: #Only saves trans_ls and cluster_ls (in id format) if the clustering result is meaningful
			utils.dict2json(raw_result_path + '_'.join(['bayesian_raw_results',suffix, str(mc_len),str(idx+1)]) + '.json', clustering_result['cluster_ls_id']) #Save clustering result (in id format) to output/raw/
			# Generate a np array, trans_ls_row, with each row contains the row to be saved to workbook
			trans_ls_np = [np.concatenate([np.asarray(pmat),col_empty],axis = 1) for pmat in clustering_result['trans_ls']] #Convert list of list to list of np arrays and add a zero column to each array
			trans_ls_row = np.concatenate(trans_ls_np, axis = 1)[:,:-1] #Concatenate list of np array and remove the last column (the last 0 column)
			for row_val in trans_ls_row: #Iterate over each row that contains (flattened) rows of trans matrices and 
				worksheet_1.append(row_val.tolist()) #Append row to worksheet
		else:
			worksheet_1.append(['','No meaningful clustering result generated!'])
	idx_meaningful = [i+1 for i, e in enumerate(cluster_len_ls) if e != 1]
	print('The number of clusters for this division are',cluster_len_ls,'and the index of meaningful clusters are',idx_meaningful)
	worksheet_1.cell(row = 1, column=3).value = str(cluster_len_ls) #Save the cluster_len_ls
	worksheet_1.cell(row = 1, column=4).value = str(idx_meaningful) #Save idx_meaningful
	worksheet_0.append([str(mc_len)+' transitions:', str(cluster_len_ls), str(idx_meaningful)]) #Write on the 'General Result' sheet
	workbook.save(workbook_path) #Save the workbook at the end of each loop (will be reopened at the beginning of next loop)
	print('Time spent on the past number of transition is',time.time() - last_time, 'and current total time taken is',time.time()-start_time)
	last_time = time.time()
###################### Test #######################
print('***********************************************************************************************')
print('Execution completed! Prior size =',sample_size,'transition number from',loop_start,'to',loop_end,'with prior type:',suffix)