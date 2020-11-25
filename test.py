import utils, func, pathlib, os
import pandas as pd


current_path = pathlib.Path(os.getcwd()) #Get the current working directory
input_data_date = '2020-11-10' #Data files are located in folders named after its running date
folder_path = str(current_path.parent)+ '/LabMachineResults/'+input_data_date #Data file path (folder name)
sorted_file_name = 'result_sorted.csv'
##################################################### Input
simplified_activities = True 
result_file =folder_path+'/'+sorted_file_name
#####################################################
NHTS_book = {1: 'Regular home activities (chores, sleep)', 2: 'Work from home (paid)', 3: 'Work', 4: 'Work-related meeting / trip', 
5: 'Volunteer activities (not paid)', 6: 'Drop off /pick up someone', 7: 'Change type of transportation', 
8: 'Attend school as a student', 9: 'Attend child care', 10: 'Attend adult care', 11: 'Buy goods (groceries, clothes, appliances, gas)',
12: 'Buy services (dry cleaners, banking, service a car, pet care)', 13: 'Buy meals (go out for a meal, snack, carry-out)',
14: 'Other general errands (post office, library)', 15: 'Recreational activities (visit parks, movies, bars, museums)',
16: 'Exercise (go for a jog, walk, walk the dog, go to the gym)', 17: 'Visit friends or relatives', 18: 'Health care visit (medical, dental, therapy)',
19: 'Religious or other community activities', 97: 'Something else', 99: 'Nothing',}

df_data = pd.read_csv(result_file, converters={'Best CJM': eval})
result_translated = pd.DataFrame()

for idx, row in df_data.iterrows():
	trip_ls = row['Best CJM'] #The bset CJMs for current trial
	trial_current = pd.concat([row]*len(trip_ls), ignore_index=True, axis = 1) #Make a dataframe with # of rows equal to number of clusters
	trial_current = trial_current.transpose()
	activity_ls = []
	for trip_ind in trip_ls:
		activity_ls.append(utils.trip_translator(input_trip = trip_ind, book=NHTS_book, single_col = simplified_activities)) #Append a activity list
	trial_current.insert(3, 'Activity List', activity_ls)
	trial_current.insert(3, 'Current Trip', trip_ls)
	
	
	result_translated = result_translated.append(trial_current)
	
	

result_translated.to_csv(folder_path+'/translated_result.csv')