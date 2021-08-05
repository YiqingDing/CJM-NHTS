import matplotlib.pyplot as plt
import numpy as np
chord_plot = __import__("matplotlib-chord")
fig = plt.figure(figsize=(6,6))
flux = np.array([[11975,  5871, 8916, 2868],
  [ 1951, 10048, 2060, 6171],
  [ 8010, 16145, 8090, 8045],
  [ 1013,   990,  940, 6907]
])

ax = plt.axes([0,0,1,1])

#nodePos = chordDiagram(flux, ax, colors=[hex2rgb(x) for x in ['#666666', '#66ff66', '#ff6666', '#6666ff']])
nodePos = chord_plot.chordDiagram(flux, ax)
ax.axis('off')
prop = dict(fontsize=16*0.8, ha='center', va='center')
nodes = ['non-crystal', 'FCC', 'HCP', 'BCC']
for i in range(4):
    ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)

plt.show()
# plt.savefig("example.png", dpi=600,
#         transparent=True,
#         bbox_inches='tight', pad_inches=0.02)
