import numpy as np
import utils, func, pandas, time,xlsxwriter, pathlib, os

last_time = time.time()
s = 21
alpha = 10 #Global precision (this val equals to 1/s for alpha_kij)
mc_len = 4 #Number of transitions
idx = 31 #Time window
t_interval = 0.5 #Default time window
##### Complete Dataset for Prior Generation #####
####Dataset generation - From start########
# raw_trip_file_complete = 'trippub.csv' #File name of the 2k data
# trip_ls_raw_complete  = func.trip_ls_input(raw_trip_file_complete,'w',save_file = False) #Generate the day trips for the complete dataset
# trip_df = func.tripls2df(trip_ls_raw_complete, t_interval)
# trip_df.to_csv('trip_df_complete.csv', index = False)
####Dataset generation########
trip_df_complete = pandas.read_csv('trip_df_complete.csv')
# mc_crop_ls_complete = func.tripdf2mcls(trip_df_complete, mc_len) #Convert trip df to a list of lists of mc
# ##################################################
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'r') #Generate the day trips for dataset
trip_df = func.tripls2df(trip_ls_raw, t_interval)
# mc_crop_ls = func.tripdf2mcls(trip_df, mc_len) #Convert trip df to mc lists of list using number of transitions
# mc_ls = mc_crop_ls[idx] 
# ##################################################
# #bayesian_main domain
# dev_prior_input = ['dev',mc_crop_ls_complete[idx]] #Form the prior input for dev prior

# cluster_ls = func.bayesian_clustering(mc_ls,alpha, s, dev_prior_input)
##################################################
# # bayesian_clustering domain
# cluster_len = len(mc_ls)
# prior_data = [dev_prior_input[1], s, 1/len(dev_prior_input[1]), cluster_len] #Input from clustering into 
# prior_ls = func.prior_generator(prior_data, type = dev_prior_input[0]) #Generate prior_ls
