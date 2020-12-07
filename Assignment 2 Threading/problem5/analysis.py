from matplotlib import pyplot as plt

threads = [1,2,4,8]
speedup = [1,1.9,2.2,3.5]
plt.plot(threads, speedup)
plt.xlabel('Number of threads')
plt.ylabel('Speedup')
plt.savefig('graph1.png')
plt.show()