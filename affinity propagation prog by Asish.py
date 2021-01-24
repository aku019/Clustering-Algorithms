from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_iris,load_digits
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
# Make Dummy Data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

#print(X)
startTime = datetime.now()

X=[]

with open("spiral.txt", "r") as filestream:
	for line in filestream:
		a=[]
		#b=[]
		currentline = line.split("\t")
	    #total = str(int(currentline[0]) + int(currentline[1]) + int(currentline [2])) + "\n"
		a=[float(currentline[0]),float(currentline[1])]
		X.append(a)


# Setup Affinity Propagation
#print(X)

X=np.array(X)
data = load_iris()
c = data['target']
X = data['data']
print(X)
centers = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
]
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=0.6)
#print(X)
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_



no_clusters = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % no_clusters)
# Plot exemplars

affinity_p_clusters = AffinityPropagation().fit_predict(X=np.array(X)[:,1:]) 

#len(np.unique(affinity_p_clusters))
#pd.DataFrame(affinity_p_clusters).plot(kind='hist',bins=np.unique(affinity_p_clusters))

#plt.show()



#plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(no_clusters), colors):
    class_members = labels == k
    print(class_members)
    cluster_center = X[cluster_centers_indices[k]]
    print(cluster_center)
    #plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.show()

print(datetime.now()-startTime)