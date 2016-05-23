__author__ = 'hxw'
#-*- coding=utf-8-*-
import numpy as np
import matplotlib.pyplot as plot
class KMeans():
	def __init__(self,train_x,k=5):
		self.train_x=train_x
		self.k=k
		self.m,self.n=train_x.shape
		self.cluster_index=np.zeros((self.m,1),dtype="int32")
		self.cluster=np.random.randn(self.k,self.n)
	def fit(self):
		cluster_change=True
		while cluster_change:
			cluster_change=False
			#step 1: assign the data to cluster center
			for i in range(self.m):
				min=np.inf
				min_index=-1
				for j in range(self.k):
					distance=np.sum((self.train_x[i]-self.cluster[j])**2)**0.5
					if distance<min:
						min=distance
						min_index=j
				if self.cluster_index[i,0]!=min_index:
					cluster_change=True
					self.cluster_index[i,0]=min_index
			#step 2:calcuate the cluster center
			for i in range(self.k):
				a=self.train_x[self.cluster_index[:,0]==i]
				if (len(a)>0):
					self.cluster[i]=a.mean(axis=0)

	def show(self):
		color=["bo","go","co","mo","yo","ko","wo"]
		for i in range(self.m):
			plot.plot(self.train_x[i,0],self.train_x[i,1],color[self.cluster_index[i,0]%7])
		plot.plot(self.cluster[:,0],self.cluster[:,1],"r*")
		print len(self.cluster)
		print len(np.unique(self.cluster_index))
		plot.show()
def loaddata(path):
	data=np.loadtxt(path)
	return data
path="D:\\pycharm_project\\KMeans\\Kmeansdata"
data=loaddata(path)
kmeans=KMeans(data,k=5)
kmeans.fit()
kmeans.show()