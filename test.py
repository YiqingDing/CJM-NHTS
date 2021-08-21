import utils, collections, csv, random, os, pathlib, ast, uuid, time, copy, itertools, sys
import pandas as pd
from math import *
import numpy as np
from bidict import bidict
import matplotlib.pyplot as plt, matplotlib.lines as ml
# from matplotlib.gridspec import GridSpec
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

dict_file_path = 'output/raw/bayesian_raw_results_test_all_1_6.json'
data = (utils.json2dict(dict_file_path)[0]) if os.path.isfile(dict_file_path) else None #Reads ini_id_dict if it exists (with the same name), else empty
print([len(i) for i in data])