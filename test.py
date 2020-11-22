import matplotlib.pyplot as plt
test_lines = (((1,1),(2,2)), ((1,2),(2,2)))
n = 1
for trip_ind in test_lines:
	print('This is line '+str(n))
	plt.plot(trip_ind[0],trip_ind[1])
	n = n+1
plt.show()

#hello world123