#!/usr/bin/env python
# coding: utf-8

# # Téléchargement des bases de données

# In[17]:


#on telecharge les bases de données issues du scrapping twitter
import pandas as pd


# In[18]:


data2021 = pd.read_csv("50 000 tweets scrappés en 2021 sur les catastrophes naturelles.csv")
data2016 = pd.read_csv("50 000 tweets scrappés en 2016 sur les catastrophes naturelles .csv", lineterminator='\n')


# # réalisation de l'analyse de sentiment sur les tweets

# In[26]:


#calcul de l'analyse de sentiment sur les tweets de l'année 2021
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment=SentimentIntensityAnalyzer()
# compound score= somme des scores positifs, négatifs et neutres qui sont normalisés entre -1 (valeur négative la + extrême) et +1 (valeur positive la plus extrême)

dict2021 = dict()
for k in data2021['Text']:
    dict2021[k] = sentiment.polarity_scores(k)

#on store les valeurs du dictionnaire dans différentes listes
list_neg = []
list_pos = []
list_neu = []
list_compound = []


for item in dict2021.values():
  list_neg.append(item["neg"])
  list_neu.append(item["neu"])
  list_pos.append(item["pos"])
  list_compound.append(item["compound"])

#on finit de créer le dataframe qui contient toutes les nouvelles valeurs du sentiment analysis
import pandas as pd
df2021new = pd.DataFrame(list(zip(list_neg, list_neu, list_pos,list_compound)), columns =['neg', 'neu','pos','compound'])
df2021new.insert(0, 'Text', dict2021.keys())
df2021new.insert(5,'date', data2021['Datetime'])
print(df2021new)


# In[49]:


#calcul de l'analyse de sentiment sur les tweets de l'année 2016

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment=SentimentIntensityAnalyzer()
# compound score= somme des scores positifs, négatifs et neutres qui sont normalisés entre -1 (valeur négative la + extrême) et +1 (valeur positive la plus extrême)

dict20 = dict()
for k in data2016['Text']:
    dict2016[k] = sentiment.polarity_scores(k)

#on store les valeurs du dictionnaire dans différentes listes
list_neg = []
list_pos = []
list_neu = []
list_compound = []


for item in dict2016.values():
  list_neg.append(item["neg"])
  list_neu.append(item["neu"])
  list_pos.append(item["pos"])
  list_compound.append(item["compound"])

#on finit de créer le dataframe qui contient toutes les nouvelles valeurs du sentiment analysis
import pandas as pd
df2016new = pd.DataFrame(list(zip(list_neg, list_neu, list_pos,list_compound)), columns =['neg', 'neu','pos','compound'])
df2016new.insert(0, 'Text', dict2016.keys())
df2016new.insert(5,'date', data2016['Datetime'])
print(df2016new)


# # Cleaning du dataset

# In[36]:


#netoyons le dataset en enlevant tous les tweets dont la neutralité est trop importante
#un tweet majoritairement neutre n'est pas d'une grande utilité dans une analyse de sentiment

df2016new = df2016new[df2016new.compound != 0.0]
df2016new = df2016new[df2016new.neu < 0.4]


df2021new = df2021new[df2021new.compound != 0.0]
df2021new = df2021new[df2021new.neu < 0.4]


# # Préparation des graphes

# In[37]:


#calcul de la positivité,négativité et neutralité moyenne des tweets de l'année 2016
import statistics 
x = statistics.mean(df2016new['pos']) 
print(x)


# In[38]:


import statistics 
x = statistics.mean(df2016new['neg']) 
print(x)


# In[39]:


import statistics 
x = statistics.mean(df2016new['neu']) 
print(x)


# In[40]:


#calcul de la positivité,négativité et neutralité moyenne des tweets de l'année 2021
import statistics 
x = statistics.mean(df2021new['pos']) 
print(x)


# In[41]:


#calcul de la positivité,négativité et neutralité moyenne des tweets de l'année 2021
import statistics 
x = statistics.mean(df2021new['neg']) 
print(x)


# In[42]:


#calcul de la positivité,négativité et neutralité moyenne des tweets de l'année 2021
import statistics 
x = statistics.mean(df2021new['neu']) 
print(x)


# In[43]:


import matplotlib.pyplot as plt

labels = 'positivity','neutrality',  'negativity'
sizes = [29.402422907488984, 39.53118313404657, 31.05838577721838]
colors = ['aqua','lightskyblue','cadetblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.savefig('PieChart01.png')
plt.title("Pourcentage de positivité, negativité et neutralité sur les catastrophes naturelles par tweets en 2016 ", fontsize = 14) 
plt.show()


# In[44]:


import matplotlib.pyplot as plt

labels = 'positivity','neutrality',  'negativity'
sizes = [28.716843783209354, 41.296041445270987, 29.98738044633369]
colors = ['aqua','lightskyblue','cadetblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.savefig('PieChart01.png')
plt.title("Pourcentage de positivité, negativité et neutralité sur les catastrophes naturelles par tweets en 2021 ", fontsize = 14) 
plt.show()


# In[ ]:


#on remarque ainsi que malgré une médiatisation croissante des catastrophes naturelles et leur multiplication sur les 5 dernières années
#on n'observe pas pour autant visiblement d'augmentation significative de l'inquiétude envers les catastrophes naturelles

#bien sûr, nous ne nions pas les biais de l'analyse de sentiment ni bien sûr les biais d'une analyse sur twitter
#en effet, twitter n'est pas un échantillion représentatif de l'ensemble des opinions des individus à travers le monde
#il est donc possible voir probable que malgré les résultats trouvés il y ait eu sur la période une augmentation de l'inquiétude des individus vis-à-vis des catastrophes naturelles

