import os, pathlib
from datetime import date

current_path = pathlib.Path(os.getcwd())
data_file = 'trippub_top2k.csv'
output_path = str(current_path.parent)+ '/LabMachineResults/'+str(date.today())
# print(current_path.parent.parent)
if not os.path.exists(output_path):
    os.makedirs(output_path)