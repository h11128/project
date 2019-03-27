
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns 
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")


# In[4]:


seeds = pd.read_csv('seeds.csv')
seeds.head()


# In[5]:


seeds.info()


# # Preliminary data analysis
# ## missing value

# In[6]:


null_data = seeds[seeds.isnull().any(axis=1)]
display(null_data)


# No missing value here  
# 
# ## outliers

# In[7]:


"""" clean outliers here by compute Z-score of each value in the column, if abs of Z score bigger than 3 than delete this row"""
data = seeds[(np.abs(stats.zscore(seeds)) < 3).all(axis=1)]
data.info()


# only deleted 2 rows so delete the outlier rows directly is a easy way to process ouelier.  
# 
# ## Correlation

# In[8]:


plt.figure(figsize = (20,7))
sns.heatmap(data.corr(), cmap ='BrBG', annot = True)
plt.title('Variables Correlation', fontsize = 18)
plt.show()


# As can be seen asym is most unrelated to other attributes.
# 
# # Data Processing
# 

# In[9]:


data.info()


# All the attribute are numerical so that don't need to be transform, outliers and missing values have been processed
# don't need to split into train and test set because this is clustering.
# 
# # Find “natural” clusters
# ## K-Means
# 

# In[10]:


X = np.array(data.iloc[:, 0:8]) 
from sklearn.cluster import KMeans
SSE = []
for K in range(2,12):
    kmeans = KMeans(init='random', n_clusters=K)
    kmeans.fit(X)
    SSE = SSE + [kmeans.inertia_]
    print(kmeans.inertia_)


# In[11]:


# Plot the SSE values we collected above
fig = plt.figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(2,12)), SSE, marker='.', markersize=10)
ax.set_title('SSE for different values of K')
plt.show()


# There is a elbow in the plot at K=3, so that appears to be the optimal number of clusters.

# In[14]:


kmeans = KMeans(init='random', n_clusters=3)
kmeans.fit(X)
y = kmeans.predict(X)
fig = plt.figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(1, 1, 1, aspect=1)
ax.scatter(X[:, 0], X[:, 1], marker='.', s=25, edgecolor='', c=y)
ax.set_title('Original Data Points')
plt.show()


# ## Hierachical
# from K-means we can see 3 cluster gives best results, so we use 3 cluster in Hierachical clustering
# Now we need to determine linkage

# In[29]:


from sklearn.cluster import AgglomerativeClustering
X = np.array(data.iloc[:, 0:8]) 
clustering = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='ward', memory=None, n_clusters=3,
            pooling_func='deprecated').fit(X)

Linkage = ["ward", "complete", "average", "single"]
j = 0
for K in Linkage:
    j = j +1
    clustering.set_params(linkage=K)
    y = clustering.fit_predict(X)
    fig = plt.figure(figsize=(12, 4), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_xlim([-0.1, 1.0])
    vertical_spacing = (j+1)*10 # extra spacing to separate the silhouettes
    ax1.set_ylim([0, len(X) + vertical_spacing])

    silhouette_avg = silhouette_score(X, y)
    print("For K = {}, the average silhouette score is {:.3f}".format(K, silhouette_avg))

    sample_silhouette_values = silhouette_samples(X, y)
    y_lower = 10
    
    for i in range(3):
        ith_cluster_silhouette_values = sample_silhouette_values[y==i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / 3)

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot")
    ax1.set_xlabel("Silhouette Score")
    ax1.set_ylabel("Cluster Label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([]) 
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    
    colors = cm.nipy_spectral(y.astype(float) / 3)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')



    ax2.set_title("Clustered Data")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle("Silhouette analysis for hierarchical clustering with linkage = {}".format(K),
                 fontsize=14, fontweight='bold')

    plt.show()


# As shown above when linkage = average the cluster gives best results

# In[50]:



X = np.array(data.iloc[:, 0:8]) 
y = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='average', memory=None, n_clusters=3,
            pooling_func='deprecated').fit_predict(X)

for i in range(1,7):
    fig = plt.figure(figsize=(12, 4), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, i], marker='.', s=25, edgecolor='', c=y)
    sting = "Feature space for the {}st feature".format(i)
    sting1 = "Feature space for the {}st feature".format(i+1)
    ax1.set_title(sting)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X[:, 0], X[:, i+1], marker='.', s=25, edgecolor='', c=y)
    ax2.set_title(sting1)
    
    

plt.show()

