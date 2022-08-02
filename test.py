import numpy as np
import pandas as pd
from math import *
from datetime import date
import matplotlib.pyplot as plt
import func, collections, utils, importlib, ujson, pathlib, os,sys

###################### Input #######################
fig, axs = plt.subplots(nrows = 1, ncols =2)
legend_anchor_bbox = (0.5,-0.075)
ax = axs[0]
x = np.arange(100)
y1 = 3*x+2
y2 = 2*x+1
y3 = 4*x+5
y4 = x+3
ax.plot(x,y1,label = '1')
ax.plot(x,y2,label='1')
ax.plot(x,y3,label='1')
ax.plot(x,y4,label='1')
leg = ax.legend(ncol=3)

renderer = fig.canvas.get_renderer()
ax_width_inch = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).width
fig_width_inch = fig.get_size_inches()[0]
fig_grid_col = ax.get_gridspec().ncols 
ax_grid_ncol = len(ax.get_subplotspec().colspan)

print('legend width',leg.get_window_extent(renderer=renderer).transformed(fig.dpi_scale_trans.inverted()).width)
print('width',fig_width_inch, ax_width_inch)

# ax.legend(bbox_to_anchor = legend_anchor_bbox, loc = 'upper center')
plt.show()