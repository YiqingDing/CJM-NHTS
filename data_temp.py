import func, utils, xlsxwriter, pandas, pathlib, os, numpy
# WARNING: This file produces results without clustering, i.e. the single transitional matrix for each time window of each # of transitions.
# The output would be a excel sheet where 
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
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset
trip_df = func.tripls2df(trip_ls_raw, t_interval) #Convert trips into df where col are time windows
s = 21
alpha = 10 #Global precision (this val equals to 1/s for alpha_kij)
# loop_iter = range(47) #Max transition number available
# Use user input for min and max number of transitions to be tested
# loop_min = int(input('Please enter the min number of transitions(inclusive): ') or 0)
# loop_max = int(input('Please enter the max number of transitions(exclusive - max 47): ') or 47)
loop_min = 0
loop_max = 47
loop_iter = range(loop_min,loop_max)
#################################################
# Write to an excel in Parent/Results/Bayesian/Bayesian_Clustering_Results.xlsx
workbook = xlsxwriter.Workbook(str(pathlib.Path(os.getcwd()).parent)+'/Results/Bayesian/Bayesian_Clustering_Results_'+os.environ.get('USER')+'.xlsx')
for i in loop_iter: #Iterate over different number of transitions
	mc_len = i+1 #Number of transitions in desired Markov chains
	mc_crop_ls, mc_title_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to mc lists of list using number of transitions
	print('MC crop list generated for mc_len=',mc_len,'!')
	worksheet_1 = workbook.add_worksheet(str(mc_len)+' Transition') #Added new sheet to record result for the specific number of transitions
	last_row_no = 1 # Last row number used for writing in worksheet_1 (reset for every mc_len)
	###################### Input #######################
	set_no = len(mc_crop_ls) #Total number of cluster sets for one day (# of time windows)
	cluster_len_ls = [] #List of cluster length
	for idx, mc_ls in enumerate(mc_crop_ls): #Iterate over all the time windows
		# Each time window contains a list of MCs
		print('--------------Clustering Starts for No.'+str(idx+1)+' out of '+str(set_no) +' sets--for MC '+str(mc_len)+'---------')
		ini_count_ls = func.mcls2mat(mc_ls, s)[1] #Get the initial count matrix
		ini_count_ls_np = [numpy.asarray(i) for i in ini_count_ls] #Convert original list of nested list to list of np array
		nmat = sum(ini_count_ls_np)
		pmat = utils.count2trans(nmat)
		
		# Saving to worksheet starts from row 1 (1st row reserved for general result)
		current_title = ['No. '+str(idx+1)] + mc_title_ls[idx] #Current title includes a number and title time windows
		worksheet_1.write_row(last_row_no,0,current_title) #Write current title (time window), at row 1, 23, etc.
		k0 = 0 #Index for row no of pmat - relative row number (reset for each time window)
		for j in range(last_row_no+1,last_row_no+s+1): #Absolute row number in excel
			# j starts from last_row_no+1 (2, 24, etc.) and ends at last_row_no+s+1 (22, 44, etc.)
			# k0 points to each row within pmat and k0 increments with j, k0\in [0,s-1]
			worksheet_1.write_row(j,0,pmat[k0]) #Write the one row for one matrix and goes to the next matrix
			k0 +=1 #Update the relative index
		last_row_no += s+1 #Update the last_row_no with the new index
		
	# Saves the file
	worksheet_1.write_row(0,0,[str(mc_len)+' transitions:']) #Write on the time window specific sheet
###################### Test #######################
workbook.close()