# WARNING: This file produces results without clustering, i.e. the single transitional matrix for each time window of each # of transitions.
# The output would be a excel sheet:
	# Each sheet contains unclustered transitional matrix of every time window for a certain transition number
	# Each sheet also contains the number of chains to be clustered by ***************
import func, utils, xlsxwriter, pathlib, os, numpy, sys, time
import matplotlib.pyplot as plt
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
last_time = time.time()
t_interval = 0.5
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset
trip_df = func.tripls2df(trip_ls_raw, t_interval) #Convert trips into df where col are time windows
s = 21
alpha = 10 #Global precision (this val equals to 1/s for alpha_kij)
# loop_iter = range(47) #Max transition number available
# Use user input for min and max number of transitions to be tested
# loop_min = int(input('Please enter the min number of transitions(inclusive - min 1): ') or 1)
# loop_max = int(input('Please enter the max number of transitions(exclusive - max 47): ') or 47)
loop_min = 1
loop_max = 47
loop_iter = range(loop_min,loop_max+1)
#################################################
# Write to an excel in Parent/Results/Bayesian/Bayesian_Clustering_Results_[USERNAME].xlsx
resultFolderPath = str(pathlib.Path(os.getcwd()).parent)+'/Results/Bayesian/'
workbook = xlsxwriter.Workbook(resultFolderPath + 'Bayesian_Clustering_Results_'+os.environ.get('USER')+'.xlsx')
cell_format = workbook.add_format({'bold': True, 'font_color': 'red'}) #Creat cell format for emphasizing
worksheet_0 = workbook.add_worksheet('WindowLabels') #Create a sheet that will save all the labels for windows
# mc_tot_num_ls = [] #Create a list of length for Markov chains (each entry is mc_num_ls)

titles_dict = {'title_sheet': 'Number of MCs for different time windows of ', 'title_win': [str(i)+' transition(s)' for i in loop_iter]}
fig_num = len(loop_iter)
ax_num = [1]*fig_num
figs, axs = func.fig_generator(fig_num, ax_num, titles_dict, tight_layout = 1, 
                               ax_kw = {'xlabel': 'Time Windows', 'ylabel': 'Number of Markov Chains in the Window', 'ylim': (0, 500), 'xlim': (0,42)},
                               suptitle_kw = {'size': 20})
axis_kw = {'tick_label_size': 10, 'axis_label_size': 15, 'fontdict': {'size': 10,'weight': 'bold'}}
#################################
# loop_iter = [0] #Test value (should be commented unless debugging)
for loop_idx, mc_len in enumerate(loop_iter): #Iterate over different number of transitions
	mc_crop_dict, mc_title_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to mc lists of list using number of transitions
	sheet_name = str(mc_len)+' Transition'
	label_title = [title_ls[0].split(' - ')[0]+' - '+title_ls[-1].split(' - ')[-1] for title_ls in mc_title_ls]
	worksheet_0.write_row(loop_idx, 0, [sheet_name]+label_title)
	# sys.exit(0)
	print('MC crop list generated for mc_len=',mc_len,'!')
	worksheet_1 = workbook.add_worksheet(sheet_name) #Added new sheet to record result for the specific number of transitions
	last_row_no = 1 # Last row number used for writing in worksheet_1 (reset for every mc_len)
	###################### Input #######################
	set_no = len(mc_crop_dict.keys()) #Total number of cluster sets for one day (# of time windows)
	mc_num_ls = [] #Create a list of number of MCs for current transition number, where each entry is the number of MCs for the corresponding time window
	for idx, (window_idx, mc_ls) in enumerate(mc_crop_dict.items()): #Iterate over all the time windows
		# Each time window contains a list of MCs, mc_ls
		print('--------------Clustering Starts for No.'+str(idx+1)+' out of '+str(set_no) +' sets--for MC of length'+str(mc_len)+'---------')
		ini_count_ls = func.mcls2mat(mc_ls, s)[1] #Get the initial count matrix
		ini_count_ls_np = [numpy.asarray(i) for i in ini_count_ls] #Convert original list of nested list to list of np array
		nmat = sum(ini_count_ls_np)
		pmat = utils.count2trans(nmat)
		
		# Saving to worksheet starts from row 1 (1st row reserved for general result)
		current_title = ['No. '+str(idx+1)] + mc_title_ls[idx] #Current title includes: A window number/index, specific time windows, and number of MCs in the window to be clustered
		worksheet_1.write_row(last_row_no,0,current_title) #Write current title (time window), at row 1, 23, etc.
		worksheet_1.write(last_row_no, len(current_title), str(len(mc_ls))+' MCs', cell_format)
		k0 = 0 #Index for row no of pmat - relative row number (reset for each time window)
		for j in range(last_row_no+1,last_row_no+s+1): #Absolute row number in excel
			# j starts from last_row_no+1 (2, 24, etc.) and ends at last_row_no+s+1 (22, 44, etc.)
			# k0 points to each row within pmat and k0 increments with j, k0\in [0,s-1]
			worksheet_1.write_row(j,0,pmat[k0]) #Write the one row for one matrix and goes to the next matrix
			k0 +=1 #Update the relative index
		last_row_no += s+1 #Update the last_row_no with the new index
		mc_num_ls.append(len(mc_ls)) #Update total number of MCs
	# Saves the file
	worksheet_1.write(0,0,str(mc_len)+' transitions:') #Write the title on the time window specific sheet
	worksheet_1.write_row(0,1, [str(mc_num_ls), str(sum(mc_num_ls)), 'Markoc Chains'], cell_format) #Write number of MCs for current sheet
	# Plot the number of MCs in current window 
	ax = axs[loop_idx]
	ax.bar(range(1,len(label_title)+1), mc_num_ls)
	# Axes and Axis properties
	ax.tick_params(axis = 'x',which = 'major' ,bottom = False, labelbottom = True, labelsize = axis_kw['tick_label_size']) #Turn off x-axis major ticks & labels
	ax.set_xticks(range(1,len(label_title)+1)) #Set x-axis minor ticks	
	ax.set_xticklabels(labels = label_title, rotation = 'vertical') #Set xtick labels and orientation
	ax.xaxis.label.set_size(axis_kw['axis_label_size']) #Set size of axis label size
	ax.yaxis.label.set_size(axis_kw['axis_label_size']) #Set size of axis label size
	rects = ax.patches #Get all the rectangles(bars) in the plot
	for rect_i, rect in enumerate(rects): #Add value to the bar
		ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height()+0.01, mc_num_ls[rect_i], ha='center', va='bottom', fontdict = axis_kw['fontdict']) #Place the prob value label

# Save all the figures to a PDF file
figFilePath = resultFolderPath+'raw/NumberofMCs.pdf'
func.fig2pdf(file_path = figFilePath, fig_num = 'all')
###################### Test #######################
workbook.close()
print('Time spent total is',time.time() - last_time)