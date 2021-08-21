import utils, collections, csv, random, os, pathlib, ast, uuid, time, copy, itertools, sys, shutil
import pandas as pd
from math import *
import numpy as np
from bidict import bidict
import matplotlib.pyplot as plt, matplotlib.lines as ml
# from matplotlib.gridspec import GridSpec
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

# dict_file_path = 'output/raw/bayesian_raw_results_test_all_1_6.json'
# data = (utils.json2dict(dict_file_path)[0]) if os.path.isfile(dict_file_path) else None #Reads ini_id_dict if it exists (with the same name), else empty
# print([len(i) for i in data])

raw_result_path = 'output/raw/' #File path to save raw result
id_dict_path = 'output/idDict' #File path to save id_dict (in Bayesian clustering)
# Removes all existing output files to avoid conflicts
# shutil.rmtree(raw_result_path,ignore_errors=True)
shutil.rmtree(raw_result_path)
shutil.rmtree(id_dict_path,ignore_errors=True)