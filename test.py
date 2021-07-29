import func, utils, statistics
t_interval = 0.5
raw_trip_file = 'trippub_top2k.csv' #File name of the 2k data
trip_ls_raw  = func.trip_ls_input(raw_trip_file,'w') #Generate the day trips for dataset
# trip_len_ls = [len(i[0]) for i in trip_ls_raw]
trip_df = func.tripls2df(trip_ls_raw, t_interval) #Convert trips into df where col are time windows
mc_len = 47
mc_crop_ls, mc_title_ls = func.tripdf2mcls(trip_df, mc_len)
