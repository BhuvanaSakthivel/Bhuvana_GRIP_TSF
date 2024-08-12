#!/usr/bin/env python
# coding: utf-8

# ## Task 1-Prediction of % score of student based on the number of study hours 

# ### Bhuvaneshwari S

# In[1]:


# Importing the reauired libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
print ("Libraries are imported successfully")


# #### Loading data and getting some insights

# In[2]:


# Importing the dataset from the device
file="D:\\Bhuvi\\Data science\\GRIP\\student_scores - student_scores.csv"
student_df=pd.read_csv(file)
student_df.head()


# In[3]:


student_df.describe()


# In[4]:


student_df.isnull().sum()


# ### Data Visualisation

# Let's look for any direct relationship between the data columns by plotting a 2D plot

# In[5]:


# 2D plot of Hourse vs scores
student_df.plot(x='Hours', y='Scores', style='*', color='maroon')  
plt.title('Relationship between study hours and percentage scored')  
plt.xlabel('Hours Studied')  
plt.ylabel('% Scored')  
plt.show()


# The relationship between the percentage scored and hourse studied is linear. Hence we can go for a simple linear regression model 

# ## Machine Learning algorithm

# Linear regression model : Linear regression uses the relationship between the data-points to draw a straight line through all them.This line can be used to predict future values.

# Let's prepare the data first by defining the predicator and target variable

# In[6]:


x=student_df.iloc[:, :-1].values
y=student_df.iloc[:, 1].values 


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=0)


# The data is split now. Next, let's train the algorthim 

# In[8]:


#importing and creating the LR object
from sklearn.linear_model import LinearRegression  
lm=LinearRegression()
lm


# In[9]:


#Let's see how the study hours affect the percentage scored by students
lm.fit(x_train, y_train)
print('Training completed')


# Let's obtain the prediction of y values for the test set

# In[10]:


yhat=lm.predict(x_test)
print(yhat)
print(lm.intercept_)
print(lm.coef_)


# In[11]:


r_sq = lm.score(x_test, yhat)
print(f"coefficient of determination: {r_sq}")


# Let's plot the regression plot 

# In[12]:


plt.scatter(x_train, y_train,color='blue', marker='*', linewidths= 2, edgecolor='green', s=200)
plt.plot(x_test, yhat,color='red', marker = 'o' )
plt.title('Relationship between study hours and percentage scored')
plt.xlabel('Study Hours')
plt.ylabel('% Scored')
plt.show()


# We have now trained our algorithm and it's time to make some predictions. To do so, we will use our test data.

# In[13]:


# Comparing actual values with predicted values.

data = pd.DataFrame({'Actual': y_test, 'Predicted': yhat})
data


# ###  What will be the percentage if a student studies for 8.5 hours?

# In[15]:


# to predict the percentage of score for particular hour, 
hours = 8.5
score= np.array([hours])
score= score.reshape(-1,1)
own_pred = lm.predict(score)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### Summary :
# 
# To predict student scores based on study hours using linear regression, we start by collecting data on study hours and corresponding scores. After organizing the data, we perform exploratory data analysis to visualize and confirm a linear relationship. We then split the data into training and testing sets, selecting study hours as the feature and scores as the target variable. Using Python's scikit-learn library, we build and train a linear regression model. Finally, we evaluate the model's performance using the testing set to predict student scores based on their study hours.
#     

# In[ ]:




