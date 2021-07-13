import func, utils, time
import os, pathlib, ast, collections
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

dataFileName = 'Bayesian_Clustering_Results_uniform.xlsx' #File that contains all meaningful results
# dataFileName = 'Bayesian_Clustering_Results_dev_3k.xlsx' #File that contains all meaningful results
rawFileName = 'Bayesian_Clustering_Results_raw.xlsx' #File that contains unprocessed results
resultFolderPath = str(pathlib.Path(os.getcwd()).parent)+'/Results/Bayesian/'
s = 21 #Number of states
labels_tot = list(utils.NHTS().values()) # Build up all the labels
threshold = 0.3 #Threshold of trans prob to be kept
plot_type = 'simulation' #Plot type: 'heatmap', 'step', 'homogeneous', 'simulation'
fig_type = 'multiple' #Figure type: 'single', 'multiple'
save_pdf = False #If saving all figures in a PDF
resultFileAffix = 'uniform'
titles_dict = collections.defaultdict(list) #Create an empty dictionary for titles
# print(resultFolderPath+resultFileAffix+'/')
##########################################################################################
dataFilePath = resultFolderPath +dataFileName
GeneralT = pd.read_excel(dataFilePath, sheet_name = 0,header = None) #Read the excel as a df
GeneralT = GeneralT[GeneralT[2]!= '[]'] #Remove rows with no meaingful result
GeneralT[2] = GeneralT[2].apply(ast.literal_eval) #Convert result col to lists from str
GeneralT[1] = GeneralT[1].apply(ast.literal_eval) #Convert result col to lists from str
GeneralT[1] = GeneralT[1].apply(lambda x: [a for a in x if a!= 1]) #Keeps only the # of clusters (remove all cluster#=1)
GeneralT[0] = GeneralT[0].apply(lambda x: int(x.replace(' transitions:',''))) #Remove extra text from the transition no
GeneralT = GeneralT.set_index(keys=0) #Use the 1st col as the index (1st col is # of transitions)
# resultDict = GeneralT.drop(labels = 1,axis = 1).T.to_dict('list', into = collections.defaultdict(list)) #Create a dictionary where keys are the transition no, vals are the meaningful time window indices for that transition no
resultNo = list(GeneralT.index) #List of all transitional no with meaningful result

# We will mix the clustered result with baseline result, i.e. fill time windows without meaningful clusters/MCs with baseline cluster (a single cluster)
baselineFilePath =  resultFolderPath+rawFileName #Get the baseline file path
processedFilePath = func.processed_data_generator(dataFilePath, baselineFilePath, resultNo, func_type = 'Read') #Get the processed file path

##########################################################################################
# resultNo = [4]
print('Current plot_type is [' + plot_type +']\nCurrent fig_type is ['+fig_type+']')
for transitionNo in resultNo: #Loop commented 
	print('Current transition number is',transitionNo)
	# transitionNo = 32 #Transition # (belong to resultNo)
	titles_dict['title_sheet'] = str(transitionNo)+' transitions' #Assign the titles for the entire sheet
	# SpecificT = pd.read_excel(processedFilePath, sheet_name = transitionNo-1,header = None) #Read the excel sheet for current transitional #
	SpecificT = pd.read_excel(dataFilePath, sheet_name = transitionNo,header = None) #Read the excel sheet for current transitional #
	dataT = SpecificT[(~SpecificT[0].isnull()) & ((SpecificT[0].apply(type) == float) | (SpecificT[0].apply(type)== int))] #Extract all the meaningful time window data (trans mat) in a single df
	windowArray = GeneralT.loc[transitionNo, 2] #Get the list of meaningful time window indices (which time windows within the current transitional # have meaningful results), this index is 1-indexed
	transCountArr = GeneralT.loc[transitionNo, 1] #Get # of transitional mat for each time window in a list
	rowRangeArr = utils.calcRow(windowArray,s, ttype = 'Result') #Get the rows in sheet where those time windows are located
	mc_sheet = collections.defaultdict(list) #Create an empty dictionary that will save corresponding MC data for this excel sheet
	for idx, windowIdx in enumerate(windowArray): #Iterate over different time windows that have meaningful results
		windowData = dataT.iloc[(idx*s):(idx+1)*s,:] #Crop out all the data for this time window (idx is window # for this transition #)
		transCount = transCountArr[idx] #Num of trans mat for this time window
		# Create empty list for both mc_data and pmat
		mc_window = []
		pmat_window = []
		# Create title for this time window
		title_idx = rowRangeArr[idx][0]-1 #Row number of the title for the current window in SpecificT
		title_row = SpecificT.iloc[title_idx,:][SpecificT.iloc[title_idx,:].notnull()] #Get the row for the title
		window_title = title_row.iloc[1].split(' - ')[0]+ ' to '+title_row.iloc[-1].split(' - ')[0]
		titles_dict['title_win'].append('Window No. '+str(windowIdx)+': '+window_title) #Assign the titles for a time window

		# Iterating over each MC within the window and plot it
		for chainNo in range(transCount): #Iterate over each Markov chain
			pmat = windowData.iloc[:,chainNo*(s+1):chainNo*(s+1)+s] #Transitional matrix for current MC
			pmat_window.append(pmat) #Append raw pmat to list (not the one after threshold)
			state_valid = utils.node_validate(pmat.to_numpy()) #Valid state within this MC
			pmat_threshold = pmat[pmat>threshold].fillna(0) #Only the nodes above threshold will count
			################
			# The following code checks if the pmat would transit to a null state which has 0 trans prob to any states
			node_valid = utils.node_validate(pmat, start_num = 0) #Get all the valid nodes (0-indexed)
			for node in node_valid: 
				transprob = pmat.to_numpy()[node]
				if 1 - sum(transprob) > 0.01:
					print('Window idx is',idx, 'chainNo is',chainNo, 'node is',node+1 ,'sum is',sum(transprob))
			################
			# Depends on the plot type, we will use different forms of data
			if plot_type == 'step' or plot_type == 'homogeneous':
				# If plot type is 'step' or 'homogeneous', we will use mc_dict keyed by edge tuple pairs and valued by trans prob
				# 'step': Treat end of edges as the next state by modifying the end states to a different set of indices (same labels still)
				# 'homogeneous': Use original states
				mc_dict = func.pmat2dict(pmat_threshold,plot_type) #Convert pmat to a dict 
				mc_window.append(mc_dict) #Append to the window
			elif plot_type == 'heatmap': 
				# If plot type is 'heatmap', we will use the transitional matrix directly
				mc_window.append(pmat_threshold)
			elif plot_type == 'simulation':
				mc_window.append(pmat) #Use original pmat for simulation (not the threshold one)
			else: #Simulation type plot has pmat already appended
				raise Exception('There is no such plot type!')
		# Note:Both entries of mc_sheet are lists, and each entry is either a mc_window or a pmat_winodw
		mc_sheet['mc_data'].append(mc_window) #Append mc_window to the list
		mc_sheet['pmat'].append(pmat_window) #Append pmat_window to the list
	##################
	last_time = time.time()
	##################
	# Get the size of plot (row and column number in plot)
	plot_size = [transCountArr,len(windowArray)] #Column# = # of time windows, row# = least common multiplier for all val in transCountArr
	# Plot everything on this excel sheet
	# func.plot_mc_sheet(mc_sheet['mc_data'],titles_dict, plot_size, plot_type = plot_type, fig_type = fig_type,save_pdf = save_pdf, 
	# 	resultFolderPath = resultFolderPath+resultFileAffix+'/', affix = resultFileAffix)
	# # Simulate all the MCs on this excel sheet
	# func.simulate_mc_sheet(mc_sheet['pmat'], n_steps = 20000, initial_state = 0, **kwargs)

	print('Time spent on this transition is',time.time() - last_time)
##########################################################################################
