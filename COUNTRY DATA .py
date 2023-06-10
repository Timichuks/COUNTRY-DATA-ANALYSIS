#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn.metrics as metrics

# machine learning
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import preprocessing


# In[2]:


# import file
df=pd.read_csv('country_data.csv')
# details of rows and column
df.head()


# In[3]:


df.shape


# In[4]:


#missing value
df.info()
df.isnull().sum()


# In[5]:


# Data Exploration
# statistical summary


# In[6]:



df.head(10)


# In[7]:


#Adjusting percentages
df['exports']=df['exports']*df['gdpp']/100
df['imports']=df['imports']*df['gdpp']/100
df['health']=df['health']*df['gdpp']/100
df.head(10)


# In[8]:


df.tail(10)


# In[9]:


df.describe()


# In[10]:


#fig = plt.figure(figsize = (12,8))
#sns.boxplot(data = Df)
#plt.show()


# In[11]:


#checking for outlier

#Q1 = Df.quantile(0.25)
#Q3 = Df.quantile(0.75)
#IQR = Q3 - Q1

#Df = Df[~((Df < (Q1 - 1.5 * IQR)) | (Df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[12]:


# showing correlation between the features
df.corr()


# In[13]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)


# In[14]:


#From the correlation it shows that exports,health,import and export are highly correlated with gdpp. 


# In[15]:


df.values


# In[16]:


df1 = df.drop("country", axis='columns')
df1


# In[17]:


#dataset clustering
distortions= []
K = range(1, 10)
#set k number of cluster for range in K
for k in K:
    kmeanModel = KMeans(n_clusters = k)
    kmeanModel.fit(df1)
#adding inertia_ to list
    distortions.append(kmeanModel.inertia_)


# In[18]:


#visualizing
plt.figure(figsize = (9,7))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method Showing the optimal K')
plt.show()


# In[19]:


#find out number of clusters
kmeans = KMeans(n_clusters = 3)


# In[20]:


X1= Three_features= df[['exports', 'gdpp', 'imports']]
X1


# In[21]:


X1.shape


# In[22]:


scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(X1)
standard_df = pd.DataFrame(standard_df, columns =['exports', 'gdpp', 'imports'])


# In[23]:


#data X(K algolrithm)
kmeans.fit(standard_df)


# In[24]:


#predict
y_kmeans = kmeans.predict(standard_df)


# In[25]:


y_kmeans.shape


# In[26]:


centers = kmeans.cluster_centers_

print(centers)


# In[27]:


y_kmeans in kmeans.labels_


# In[28]:


df['cluster_country'] = y_kmeans


# In[29]:


plt.scatter(standard_df['gdpp'], standard_df['imports'], c=df['cluster_country'],cmap= 'rainbow')


# In[30]:


u_labels = np.unique(y_kmeans)


# In[31]:


np.unique(y_kmeans)


# In[32]:


for i in u_labels:
    print(i)


# In[33]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)
 
#plotting the results:
for i in u_labels:
    plt.scatter(X1.iloc[y_kmeans == i , 0] , X1.iloc[y_kmeans == i , 1], label = i)
plt.xlabel('exports')
plt.ylabel('gdpp')
plt.legend()
plt.show('cluster.png')


# In[34]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)
 
#plotting the results:
#for i in u_labels:
    #plt.scatter(X2.iloc[y_kmeans == i , 0] , X.iloc[y_kmeans == i , 2],  label = i)
#plt.xlabel('imports')
#plt.ylabel('gdpp')
#plt.legend()
#plt.show()


# In[35]:


np.unique(y_kmeans)


# In[36]:


y_kmeans


# In[37]:


X1['country_clusters']=y_kmeans


# In[38]:


X1.head()


# In[39]:


X1['c'] = df.country


# In[40]:


X1.head()


# In[41]:


#country clusters that represent 0
cluster1 = X1.loc[X1.country_clusters==0]


# In[42]:


cluster1['c'].unique()


# In[43]:


#country clusters that represent 1
cluster2 = X1.loc[X1.country_clusters==1]


# In[44]:


cluster2['c'].unique()


# In[45]:


#country clusters that represent 0
cluster3 = X1.loc[X1.country_clusters==2]


# In[46]:


cluster3['c'].unique()


# In[47]:


X1['country_clusters'].value_counts()


# In[48]:


Country_cluster_exports=pd.DataFrame(X1.groupby(["country_clusters"]).exports.mean())
Country_cluster_gdpp=pd.DataFrame(X1.groupby(["country_clusters"]).gdpp.mean())
Country_cluster_import=pd.DataFrame(X1.groupby(["country_clusters"]).imports.mean())


# In[49]:


df1 = pd.concat([Country_cluster_exports, Country_cluster_gdpp, Country_cluster_import], axis=1)


# In[50]:


df1


# In[51]:


X2 = df[['exports', 'imports', 'gdpp', 'health', 'income', 'life_expec']]


# In[52]:


X2


# In[53]:


scaler = preprocessing.StandardScaler()
standard_df2 = scaler.fit_transform(X2)
standard_df2 = pd.DataFrame(standard_df2, columns =['exports', 'gdpp', 'imports', 'health', 'income', 'life_expec'])


# In[54]:


#run the Kmeans algorithm for the data X:
kmeans.fit(standard_df2)


# In[55]:


#predict which cluster each data point X belongs to:
y2_kmeans = kmeans.predict(standard_df2)


# In[56]:


centers2 = kmeans.cluster_centers_
#print(centers)
print(centers2)


# In[57]:


y2_kmeans in kmeans.labels_


# In[58]:


u2_labels = np.unique(y2_kmeans)


# In[59]:


np.unique(y2_kmeans)


# In[60]:


for i in u2_labels:
    print(i)


# In[61]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y_kmeans)
 
#plotting the results:
for i in u2_labels:
    plt.scatter(X2.iloc[y_kmeans == i , 0] , X2.iloc[y_kmeans == i , 1], label = i)


plt.legend()
plt.show('cluster1.png')


# In[62]:


y2_kmeans


# In[63]:


X2['country_clusters']=y2_kmeans


# In[64]:


X2.head()


# In[65]:


X2['c'] = df.country


# In[66]:


group1 = X2.loc[X2.country_clusters==0]


# In[67]:


group1['c'].unique()


# In[68]:


group2 = X2.loc[X2.country_clusters==1]


# In[69]:


group2['c'].unique()


# In[70]:


group3 = X2.loc[X2.country_clusters==2]


# In[71]:


group3['c'].unique()


# In[72]:


X2['country_clusters'].value_counts()


# In[73]:


X2.groupby(['country_clusters']).mean()


# In[74]:


X4 = df[['child_mort','exports', 'imports', 'gdpp', 'health', 'income', 'inflation', 'life_expec', 'total_fer']]


# In[75]:


scaler = preprocessing.StandardScaler()
standard_df3 = scaler.fit_transform(X4)
standard_df3 = pd.DataFrame(standard_df3, columns =['child_mort','exports', 'imports', 'gdpp', 'health', 'income', 'inflation', 'life_expec', 'total_fer'])


# In[76]:


#run the Kmeans algorithm for the data X:
kmeans.fit(standard_df3)


# In[77]:


#predict which cluster each data point X belongs to:
y4_kmeans = kmeans.predict(standard_df3)


# In[78]:


centers4 = kmeans.cluster_centers_
#print(centers)
print(centers4)


# In[79]:


u4_labels = np.unique(y4_kmeans)


# In[80]:


for i in u2_labels:
    print(i)


# In[81]:


X4['country_clusters']=y4_kmeans


# In[82]:


X4['c'] = df.country


# In[83]:


X4['country_clusters'].value_counts()


# In[84]:


grp1 = X4.loc[X4.country_clusters==0]


# In[85]:


grp1['c'].unique()


# In[86]:


grp2 = X4.loc[X4.country_clusters==1]


# In[87]:


grp2['c'].unique()


# In[88]:


grp3 = X4.loc[X4.country_clusters==2]


# In[89]:


grp3['c'].unique()


# In[92]:


X4.groupby(['country_clusters']).mean()


# In[90]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(y4_kmeans)
 
#plotting the results:
for i in u2_labels:
    plt.scatter(X4.iloc[y_kmeans == i , 0] , X4.iloc[y_kmeans == i , 1], label = i)


plt.legend()
plt.show('cluster2.png')


# In[91]:


#visuaizing all clusters
fig= plt.figure(figsize= (10,9))
sns.scatterplot(x='gdpp',y='income',hue='country_clusters',legend='full',data=X4)
plt.xlabel('GDPP',fontsize=10)
plt.ylabel('Income',fontsize=10)
plt.title('GDPP. vs income per person')
plt.show()


# In[ ]:




