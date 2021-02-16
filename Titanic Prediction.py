#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[179]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

traind=open(r"C:\Users\NUTHETI SRI LAASYA\Documents\Datasets\Titanic\train.csv")
testd=open(r"C:\Users\NUTHETI SRI LAASYA\Documents\Datasets\Titanic\test.csv")

train=pd.read_csv(traind)
test=pd.read_csv(testd)
name=train.Name
train.head()


# In[127]:


train.shape


# In[128]:


train.isnull().sum()


# ## Cleaning Data

# In[129]:


train = train.drop('Name', axis=1,)
train = train.drop('Ticket', axis=1,)
train = train.drop('Fare', axis=1,)
train = train.drop('Cabin', axis=1,)


# In[130]:


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')


# In[131]:


train['Family'] = train['SibSp'] + train['Parch'] + 1


# In[132]:


train = train.drop('SibSp', axis=1,)
train = train.drop('Parch', axis=1,)


# In[133]:


train["Age"] = train["Age"].astype(np.float16)
train["Age"] = (train["Age"].fillna(train["Age"].median()))
embarkedmode=(train["Embarked"].mode())
train["Embarked"] = train["Embarked"].fillna("S")


# In[134]:


train.isnull().sum()


# In[135]:


#train["Age"].describe()
train.Age


# In[136]:


train["Adult"] = 0


# In[137]:


train["Adult"][train["Age"] >= 18] = 1


# In[138]:


print ("Passengers more than 18 yrs old: ",str(len(train[train["Age"] >= 18])))
print ("Number of Adults: ",str(len(train[train["Adult"] >= 1])))


# In[139]:


train = train.drop('Age', axis=1,)


# ## Data Visualization

# ### Figure depicting co-relation between all the features

# In[140]:


plt.subplots(figsize = (15,8))
sns.heatmap(train.corr(), annot=True,cmap="PiYG")
plt.title("Correlations Among Features", fontsize = 20);


# In[141]:


def visualize (parameter1, parameter2):
    ref = train[[parameter1, parameter2, 'PassengerId']]
    ref_p = ref.pivot_table(index=[parameter1], columns=[parameter2], aggfunc=np.size, fill_value=0)
    
    p_chart = ref_p.plot.bar()
    for p in p_chart.patches:
        p_chart.annotate(str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.01))
    
    return ref_p
    return p_chart


# In[142]:


visualize("Survived","Pclass")


# In[143]:


visualize("Survived","Sex")


# In[144]:


visualize("Survived","Embarked")


# In[145]:


visualize("Survived","Family")


# In[146]:


visualize("Survived","Adult")


# ##### Feel free to represent and visualize the obtained data as you like in other ways!

# ## Finding features which are to be considered using SelectKBest

# In[147]:


dataframe = train.filter(['Pclass','Sex','Embarked','Family','Adult'], axis=1)

df = dataframe


# In[148]:


dataframe1 = train['Survived']

df_1 = dataframe1


# In[149]:


df["Embarked"][train["Embarked"] == "S"] = 1
df["Embarked"][train["Embarked"] == "C"] = 2
df["Embarked"][train["Embarked"] == "Q"] = 3


# In[150]:


df["Sex"][train["Sex"] == "male"] = 1
df["Sex"][train["Sex"] == "female"] = 2


# In[151]:


select = SelectKBest(f_classif, k='all')
fit = select.fit(df, df_1)
score = fit.scores_.round(3)
values = -np.log10(fit.pvalues_).round(3)


# In[152]:


features = list(df.columns.values)
features_final = select.get_support([fit])
features_final


# In[153]:


temp = [ ]

for i in features_final:
    temp.append({'Feature':features[i], 'Value':values[i], 'Score': score[i]  })
    
selected = pd.DataFrame(temp)


# In[154]:


selected = selected.sort_values(by='Score', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')


# In[155]:


selected = selected.set_index('Feature')


# In[156]:


selected


# #### From the values above, we can just consider the values- Sex, Pclass, and Adult

# In[157]:


df = df.drop('Embarked', axis=1,)
df = df.drop('Family', axis=1,)


# ## Training and testing the model

# In[158]:


ftrain, ftest, ltrain, ltest = train_test_split(df, df_1, test_size=0.3, random_state=42)


# In[159]:


ftrain.shape


# In[160]:


ftest.shape


# In[161]:


ltrain.shape


# In[162]:


ltest.shape


# In[163]:


ftrain["Sex"] = ftrain["Sex"].astype(np.int64)


# In[164]:


ltrain.dtypes


# In[165]:


kweights = ['uniform','distance']
krange = list(range(1,10))
scores = {}
scores_list = []
for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn.fit(ftrain, ltrain)
    lpred=knn.predict(ftest)
    scores[k]= metrics.accuracy_score(ltest, lpred)
    scores_list.append(metrics.accuracy_score(ltest, lpred))


# In[166]:


scores_list


# In[167]:


scores

