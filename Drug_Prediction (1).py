#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


df=pd.read_csv("drug200.csv")


# In[7]:


df[0:5]


# In[13]:


x=df[['Age','Sex','BP','Cholesterol','Na_to_K']].values
y=df['Drug']


# In[22]:


from sklearn import preprocessing 
x_sex=preprocessing.LabelEncoder()
x_sex.fit(['M','F'])
x[:,1]=x_sex.transform(x[:,1])
x_BP=preprocessing.LabelEncoder()
x_BP.fit(['HIGH','LOW','NORMAL'])
x[:,2]=x_BP.transform(x[:,2])
x_Chol=preprocessing.LabelEncoder()
x_Chol.fit(['HIGH','NORMAL'])
x[:,3]=x_Chol.transform(x[:,3])
x[0:5]


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
drugtree=DecisionTreeClassifier(criterion="entropy",max_depth=4)
drugtree


# In[28]:


drugtree.fit(x_train,y_train)


# In[31]:


k=drugtree.predict(x_test)
print(k[0:5])
print(y_test[0:5])


# In[32]:


from sklearn import metrics
print("The accuracy is ",metrics.accuracy_score(y_test,k))


# In[36]:


get_ipython().system('conda install -c conda-forge pydotplus -y')
get_ipython().system('conda install -c conda-forge python-graphviz -y')


# In[37]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugtree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:




