#!/usr/bin/env python
# coding: utf-8

# # TASK 6 : Decision tree classifier for Iris dataset Class prediction 

# ## Bhuvaneshwari S

# In[1]:


# Importing the reauired libraries 
import pandas as pd
import numpy as n
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report 
from sklearn import metrics

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


file="D:\\Bhuvi\\Data science\\GRIP\\Iris (1).csv"
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

# In[13]:


# Let's see the correlation between the attributes
corr = iris_df.corr()
round(corr,3)


# Here, +1 means variables are correlated, -1 inversely correlated. The petal length and petal width are highly correlated, as well as the Petal width and sepal length have a good correlation.

# # Decision Tree modeling

# ### Criterion : Gini - The Gini Index is the additional approach to dividing a decision tree. The Gini index lies between 0 to 0.5.
# 
# The internal working of Gini impurity is also somewhat similar to the working of entropy in the Decision Tree.
# Purity and impurity in a junction are the primary focus of the Entropy and Information Gain framework.
# The Gini Index, also known as Impurity, calculates the likelihood that somehow a randomly picked instance would be erroneously cataloged. 

# In[14]:


X=iris_df.drop("Species",axis=1)
Y=iris_df["Species"]


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[16]:


X_train.shape


# In[17]:


Y_train.shape


# In[18]:


dtc_G = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
dtc_G.fit(X_train,Y_train)


# In[19]:


# Predict the Test set results with criterion gini index
Y_predict_G = dtc_G.predict(X_test)


# Constructing confusion matrix

# In[20]:


from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(Y_test, Y_predict_G)))


# In[21]:


# Let's compare the sets fitting accuracy
Y_predict_train_G = dtc_G.predict(X_train)

Y_predict_train_G


# In[22]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(Y_train, Y_predict_train_G)))


# In[23]:


#Check for overfitting and underfitting
print('Training set score: {:.4f}'.format(dtc_G.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(dtc_G.score(X_test, Y_test)))


#  These two values are quite comparable. So, there is no sign of overfitting.

# ### criterion=entropy-entropy helps us to build an appropriate decision tree for selecting the best splitter. Entropy can be defined as a measure of the purity of the sub-split. Entropy always lies between 0 to 1. 

# In[24]:


dtc_E = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtc_E.fit(X_train,Y_train)


# In[25]:


# Predict the Test set results with criterion gini index
Y_predict_E = dtc_E.predict(X_test)


# In[26]:


print('Model accuracy score with criterion entropy index: {0:0.4f}'. format(accuracy_score(Y_test, Y_predict_E)))


# In[27]:


# Let's compare the sets fitting accuracy
Y_predict_train_E = dtc_E.predict(X_train)

Y_predict_train_E


# In[28]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(Y_train, Y_predict_train_E)))


# In[29]:


#Check for overfitting and underfitting
print('Training set score: {:.4f}'.format(dtc_E.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(dtc_E.score(X_test, Y_test)))


# These two values are quite comparable. So, there is no sign of overfitting.

# The accuracy scores indicates the models accurate prediction of the classs. However, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors made by the models. In such cases, Confusion matrix can be used.
# 
# ## Confusion matrix
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

# In[30]:


cm = confusion_matrix(Y_test, Y_predict_E)
print('Confusion matrix\n\n', cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=dtc_E.classes_)
disp.plot()

plt.show()


# Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. I have described these terms in later.

# In[31]:


print(classification_report(Y_test, Y_predict_E))


# In[32]:


print (Y_predict_E [0:5])
print (Y_test [0:5])


# ## Decision Tree visualisation

# In[33]:


fn=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
cn=['setosa','versicolor','virginica']

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

tree.plot_tree(dtc_E, feature_names = fn, class_names = cn, filled = True);


# # Summary 
# In this project, I developed a Decision-Tree Classifier to predict the species of iris flowers, using both the Gini index and entropy as criteria. Both models demonstrated strong performance, achieving an accuracy of 0.9556. For the model using the Gini index criterion, the training accuracy was 0.9810, and the test accuracy was 0.9556, indicating no signs of overfitting. Similarly, the entropy-based model showed the same training and test accuracies, mirroring the results of the Gini index model. The consistency between the training and test accuracies in both models suggests robust generalization, possibly influenced by the relatively small size of the dataset. Additionally, the confusion matrix and classification report confirm the model's strong performance.

# In[ ]:




