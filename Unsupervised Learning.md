Sometimes we are unable to provide nicely labeled data, in this case we would want a model that can label given data without any help, hence the term "Unsupervised".

# How it works
the main idea of unsupervised learning is to cluster given data into separate clusters, and try to reduce the dimensions of the data, reducing the features to bare minimum such that these features provide meaningful interpretability, this also help to reduce noise in the data.

# Clustering
To cluster unlabeled data, we will need to first understand what defines a cluster
## Clusters 
Clusters refer to groups of data points that share similar characteristics or are closer to each other in a given feature space
![[Pasted image 20241118062638.png]]
## Centroids
A centroid is a point that represents the center of a cluster and acts as a reference for grouping data points.
![[Pasted image 20241118062725.png]]
This can be calculated by:
$$\mu=\frac{1}{m}\sum^m_{i=1}x^{(i)}$$
where m is the number of points, $\mu$ is the centroid.
## K-means method
The K means method finds K clusters in a dataset. 
The steps are as follows:

Randomly initialize 𝐾 centroids: 
	$\mu_1 , … , \mu_𝐾$ 
Repeat until convergence: 
	For 𝑖 = 1, … , 𝑚: 
		$𝑐^{(𝑖)}$ ← index of cluster centroid $\mu_1 , … , \mu_𝐾$  closest to $x^{(i)}$ 
	For 𝑘 = 1, … ,𝐾: 
		$\mu_k$ ← centroid of data points $x^{(i)}$  assigned to cluster $k$

This basically means:
randomly select k centroids 
	while changes are made:
		classify data base on data
		find centroids of these data
		reclassify the data based on the new centroids

## Local optima
sometimes the method will result in a local optimal solution (right) instead of a global optimal solution (left):![[Pasted image 20241118063739.png]]
Therefore there is a need to measure the "goodness" of the resulted classification using a loss function (using MSE):
$$J(c^{(1)},c^{(2)},...,c^{(m)},\mu_{1},\mu_{2},...,\mu_{k}) = \frac{1}{m}\sum_{i=1}^m||x^{(i)}-\mu_{c^{(i)}}||^2$$
## elbow method
To choose the best number of clusters, we will choose it base on the elbow method. The goal is to identify the k value where adding more clusters does not significantly improve the clustering performance, indicated by a "bend" or "elbow" in the evaluation metric. We will plot the graph of loss against the number of K, choosing the last point of inflection.
![[Pasted image 20241118064222.png]]
However, this is only a heuristic method. In real life, there are chances where the model have no elbow or multiple elbows, therefore this method might not be able to provide insightful information about how good the k value is.

## K-Medoids
Instead of choosing random centroid, we can pick a random datapoint as the starting point. For each iteration, instead of using the calculated centroid as the next centroid, we choose the nearest datapoint as the next centroid.

K-medoids are less sensitive to outliers, requires higher computational cost, but is better with data in non-Euclidean space (e.g. graph data).

# Hierarchical Clustering
sometimes we cannot decide on a fixed number of clusters. We want a hierarchy of clusters, that can show multiple amounts of clusters as required. this is where we will have Hierarchical Clustering. 
![[Pasted image 20241118065550.png]]
the algorithm is very simple:

we make every point into its own cluster
Loop (until all points are in one cluster): 
	Find a pair of cluster that is “nearest”, merge them together

# Dimension reduction
In machine learning, we always want to reduce the number of features as number of samples to learn a hypothesis class increases exponentially with the number of features. 

Sometimes its easy to reduce dimensions as some features are obviously not important:![[Pasted image 20241118070133.png]]
However, in some situation, all features are equally important. this is where we will change the basis of the graph:
![[Pasted image 20241118070251.png|300]]![[Pasted image 20241118070310.png|300]]
Reduce the non important features:
![[Pasted image 20241118070442.png]]
We can also perform these steps in reverse, to reconstruct the original dimensions

## Single Value Decomposition (SVD)
This basically describes the process of SVD. In SVD, we decomposes any given matrices into 3 different matrices:
$$X=U\Sigma V^T$$
This 3 matrices represents:
- $U$ - left-singular vectors. this is new basis, basically the "components" that can combine with each other to form back the original matrix
- $\Sigma$ - singular value. basis importance, this value shows how significant data in $U$ and $V$ is in the original matrix $X$
- $V^T$ - right-singular vectors. combiner, how much of the "components" are required to form back the original matrix 

these 3 matrices are all ordered from most important to least important, based on the values in $\Sigma$, where highest value is on top, lowest is on the bottom.
![[Pasted image 20241118070814.png]]

Since this process is a very calculation intensive process, it would be beneficial if we can reduce the number of calculations. instead of finding a SVM with $m$ singular value, we can find a SVD with $r$ singular values, that approximates to our original matrix, with a tolerable loss of information.
![[Pasted image 20241118072548.png]]
This means that we can now reduce the number of dimensions easily, with some tolerable loss of data. In this sense, we can use SVD as encoder-decoder machine, that help us reduce the number of dimensions and restoring it to the original dimensions.
![[Pasted image 20241118072901.png]]

## Principal Component Analysis (PCA)
How do we determine how much data is tolerable? We can measure it using variance. Higher variance means that there are more statistical variation in our data, which means that that "component" is more important. Using this idea, we can perform SVD base on the covariance of the data.
Since $Var(x) = E[(x-mean(x))^2]$ and $Cov(x,y) = E[(x-mean(x))(y-mean(y))]$ we can deduce that:
$$\begin{align} &\text{the sample mean of x: }&\bar x = \frac{1}{m}\sum^m_{i=1}x^{(i)}\\
&\text{the mean centered data }&\hat x: \hat x=x^{(i)}-\bar x\\
&\text{The covarience of the data: }&Cov(X)=\widehat X \widehat X^T\end{align}$$
We can then calculate the decomposition:
![[Pasted image 20241118074538.png]]
and find out how many percent of the original variance is reserved by calculating:
$$\left(\frac{\sum^r_{i=1}\sigma_i^2}{\sum^r_{i=1}\sigma_i^2}\right)\times100\%$$
Using PCA, it helps us compress the data, which saves us space and speed up calculations. this also can help us visualize our data if we compress it to $r=2$ or $r=3$.



