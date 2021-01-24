import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_iris,load_digits
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd 
from sklearn.cluster import k_means
from scipy.spatial import distance  
from sklearn.datasets import load_sample_images



from datetime import datetime
startTime = datetime.now()
X=[]

with open("aggregation.txt", "r") as filestream:
	for line in filestream:

		a=[]
		currentline = line.split("\t")
		a=[float(currentline[0]),float(currentline[1]),float(currentline[2])]
		X.append(a)


#data = load_breast_cancer()

#noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      #noise=.05)

#df = pd.read_csv("E:/Anaconda/dataset2.csv")        

#df = df.dropna()
	# print df
#x1 = df.copy()
#del x1['isLegendary']
#del x1['Effective To Date']
#x4 = pd.get_dummies(x1)

#X=[]
#X=np.array(x4)
X=np.array(X)




for i in range(1,2):
	bandwidth = estimate_bandwidth(X, quantile=.1,
                               n_samples=1000)
	print(bandwidth)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_
	n_clusters_ = labels.max()+1
	#vor = Voronoi(cluster_centers)
	#fig = voronoi_plot_2d(vor)
	#plt.show()
    
#%% Plot result
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1],
             'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

print(datetime.now()-startTime)