import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
#k_means
from datetime import datetime
from sklearn.datasets import load_iris,load_digits
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
startTime = datetime.now()






X=[]

with open("spiral.txt", "r") as filestream:
	for line in filestream:
		a=[]
		currentline = line.split("\t")
		a=[currentline[0],currentline[1]]
		X.append(a)


X=np.array(X,dtype=float)



#X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True)
brc.fit(X) 
Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None,   threshold=0.5)
print(brc.fit_predict(X, y=None))

Y=brc.predict(X)
#array([0, 0, 0, 1, 1, 1])
#print(Y)

labels = brc.labels_
n_clusters_ = labels.max()+1
cluster_centers = brc.subcluster_centers_ 

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


