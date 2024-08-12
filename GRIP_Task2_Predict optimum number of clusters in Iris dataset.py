#!/usr/bin/env python
# coding: utf-8

# # TASK 2 : Predicting the optimum number of clusters in Iris dataset and visualise it

# ## Bhuvaneshwari S

# In[1]:


# Importing the reauired libraries 
import pandas as pd
import numpy as n
import matplotlib.pyplot as plt  
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
print ("Libraries are imported successfully")


# In[2]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# ### Loading Data and visualise it

# In[3]:


file="D:\\Bhuvi\\Data science\\GRIP\\Iris.csv"
iris_df=pd.read_csv(file)
iris_df.head()


# ### Process the data and gain statistical information

# In[4]:


# Let's drop the id rename the columns first
iris_df.drop(columns=["Id"], inplace=True)
iris_df.rename(columns={'SepalLengthCm': 'Sepal_Length', 'SepalWidthCm': 'Sepal_Width', 'PetalLengthCm' :'Petal_Length','PetalWidthCm': 'Petal_Width'}, inplace=True)
iris_df.head()


# In[5]:


iris_df.value_counts()


# In[6]:


iris_df.describe()


# ## Class Distribution
# Let's look at the number classes in the dataset

# In[7]:


print(iris_df.groupby('Species').size())


# ### Description: 
# 
# The iris dataset, consists of 3 different species' of irises’ (Setosa y=0, Versicolour y=1, and Virginica y=2) petal and sepal length, stored in a 150x4 numpy.ndarray
# 
# The rows contains the samples and the columns contains the attributes namely Sepal Length, Sepal Width, Petal Length and Petal Width.

# ### DATA VISUALISATION
# 
# With a basic idea about the data, now let's extend that with some visualizations.
# 
# Will be using two types of plots:
# 
# Univariate plots to understand each attribute.
# Multivariate plots to better understand the relationships between attributes.

# In[8]:


fig, axes = plt.subplots(2,2, figsize=(16,9))

sns.boxplot(x='Species', y='Sepal_Length', data=iris_df, orient='v', ax=axes[0,0])
sns.boxplot(x='Species', y='Sepal_Width', data=iris_df, orient='v', ax=axes[0,1])
sns.boxplot(x='Species', y='Petal_Length', data=iris_df, orient='v', ax=axes[1,0])
sns.boxplot(x='Species', y='Petal_Width', data=iris_df, orient='v', ax=axes[1,1])


# From the subplots, it can be inferred that, 
# Setosa species have smaller dimensions, are less distributed and have some outliers in the Petal length and width.
# Versicolor species are distributed in an average manner and has average dimensions.
# Virginica species are highly distributed with a larger number of values and dimensions.

# In[9]:


iris_df.hist(color='red')
plt.show()


# Scatter plot can be used to understnad the relation between the attributes. In this section, let's analyse the attributes and identify the correlation between them.

# In[10]:


plt.figure(figsize=(14,6))

plt.title('Comparison of various species based on Sepal length and width')

sns.scatterplot(x=iris_df['Sepal_Length'], y=iris_df['Sepal_Width'], hue=iris_df['Species'], s=50)


# From the plot above, we can say;
# 
# Iris-setosa species have smaller sepal lengths and higher sepal width.
# Iris-versicolor species lies in the middle for both its sepal length and sepal width.
# Iris-virginica species have higher sepal length and smaller sepal width.

# In[11]:


plt.figure(figsize=(14,6))

plt.title('Comparison of various species based on Petal length and width')

sns.scatterplot(x=iris_df['Petal_Length'], y=iris_df['Petal_Width'], hue=iris_df['Species'], s=50)


# From the plot above, we can say;
# 
# Iris-setosa species have the smallest petal length and petal width.
# Iris-versicolor species have average petal length and petal width.
# Iris-virginica species have the highest petal length and petal width.

# In[12]:


# Let's see the correlation between the attributes
corr = iris_df.corr()
round(corr,3)


# Here, +1 means variables are correlated, -1 inversely correlated. The petal length and petal width are highly correlated, as well as the Petal width and sepal length have a good correlation.

# ### Data Modeling

# K-Means Clustering Algorithm
# 
# K-Means is an unsupervised machine learning algorithm used to partition a dataset into K distinct, non-overlapping clusters based on the similarity of data points. The goal is to group data points such that points in the same cluster are more similar to each other than to those in other clusters. 
# 
# The number of clusters K is not determined by the algorithm and must be specified beforehand. Choosing an appropriate K is crucial and can be guided by methods like the Elbow Method. 
# 
# Elbow Method: Plot the sum of squared distances from each point to its assigned cluster centroid for different values of K. The optimal K is often at the "elbow" point, where the reduction in the sum of squared distances slows down significantly. 

# In[13]:


# Let's plot the elbow method plot to determine the K-Value from it
X=iris_df.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,12):
    k_means = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)

# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 12), wcss, marker ='*', color='red')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()




# In[14]:


k_means_labels = k_means.labels_
k_means_labels


# In[15]:


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# From the elbow plot the optimum number of clusters can be considered as 3. Now k means can be applied to the dataset

# In[16]:


k_means = KMeans(n_clusters = 3, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
Yk_means = k_means.fit_predict(X)

#Visualising the clusters - On the first two columns
plt.scatter(X[Yk_means == 0, 0], X[Yk_means == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[Yk_means == 1, 0], X[Yk_means == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[Yk_means == 2, 0], X[Yk_means == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# ### Summary : 
# 
# In the Iris dataset, the three species—Setosa, Versicolor, and Virginica—exhibit distinct characteristics in their petal and sepal measurements, which can be visualized and analyzed using box plots and other statistical tools.
# 
# Setosa Species:
# 
# The Setosa species is notable for having some outliers in both petal length and petal width. These outliers represent data points that differ significantly from the rest of the data, indicating some unusual variations in certain flowers.
# Setosa generally has smaller dimensions for both petal and sepal measurements compared to the other species. The distribution of Setosa's features is less spread out, meaning the data points are more clustered around a central value.
# The limited spread and smaller feature size make Setosa distinct and easily separable from the other species, especially when visualizing the data.
# Versicolor Species:
# 
# Versicolor species exhibit a more average distribution in terms of their features. The petal and sepal measurements are neither too small nor too large, falling in the middle range of the dataset.
# The distribution of Versicolor's features is moderately spread out, indicating some variation in the measurements but not as pronounced as in Virginica. This middle-ground positioning makes Versicolor an interesting species for comparing with both Setosa and Virginica.
# Virginica Species:
# 
# Virginica species are characterized by having the largest dimensions in petal and sepal measurements. The features are highly distributed, with a wide range of values, indicating significant variability among the samples.
# The large spread in Virginica's measurements suggests a high degree of variability, which could be attributed to the species' diverse environmental adaptations or genetic variations.
# Virginica's larger and more distributed features make it the most variable species in the dataset, providing a strong contrast to the more clustered Setosa species.
#     

# In[ ]:




