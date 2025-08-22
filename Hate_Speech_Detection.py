#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[4]:


dataset = pd.read_csv("twitter.csv.zip")


# In[5]:


dataset


# In[6]:


dataset.head(10)


# In[7]:


dataset.tail(10)


# In[8]:


dataset.describe()


# In[9]:


dataset.isnull()


# In[10]:


dataset.isnull().sum()


# In[11]:


import re
import nltk
import string


# In[20]:


from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))


# In[22]:


#import stemming
stemmer = nltk.SnowballStemmer("english")


# In[23]:


dataset["labels"] = dataset["class"].map({0: "Hate Speech" ,1: "Offensive Langauge" , 2: "Normal"})


# In[24]:


data = dataset[["tweet", "labels"]]


# In[25]:


data


# In[27]:


def clean_data(text):
    # 1. Convert to lowercase and ensure string
    text = str(text).lower().strip()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove words containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text)

    # 4. Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # 5. Remove newline characters and extra spaces
    text = re.sub(r'\s+', ' ', text)

    # 6. Remove stopwords and apply stemming in one pass
    words = [stemmer.stem(word) for word in text.split() if word not in stopwords]

    return " ".join(words)


# In[28]:


data["tweet"] = data["tweet"].apply(clean_data)


# In[29]:


data


# In[30]:


X = np.array(data["tweet"])
Y = np.array(data["labels"])


# In[31]:


#CounterVectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[32]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[34]:


X


# In[35]:


X_train


# In[36]:


Y


# In[37]:


Y_train


# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


data1 = DecisionTreeClassifier()
data1.fit(X_train, Y_train)


# In[40]:


Y_pred = data1.predict(X_test)


# In[41]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[42]:


cm


# In[43]:


Y_pred


# In[44]:


X_test


# In[45]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


#HeatMap with sns(seaborn)
sns.heatmap(cm, annot=True, fmt=".1f", cmap="rocket_r")


# In[47]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(Y_pred, Y_test)


# In[55]:


sample = "let's unite and kill all the people who are...."
sample = clean_data(sample)


# In[56]:


sample


# In[57]:


data2 = cv.transform([sample]).toarray()


# In[58]:


data2


# In[48]:


# Calculate precision, recall, and f1 with multiclass averaging
pre = precision_score(Y_test, Y_pred, average='weighted')  # Use 'micro' or 'macro' as needed
rec = recall_score(Y_test, Y_pred, average='weighted')  # Use 'micro' or 'macro' as needed
f1 = f1_score(Y_test, Y_pred, average='weighted')  # Use 'micro' or 'macro' as needed


# In[49]:


# Print the metrics
acc, pre, rec, f1


# In[59]:


#Project Hate_Speech_detection '''COMPLETED'''


# In[ ]:




