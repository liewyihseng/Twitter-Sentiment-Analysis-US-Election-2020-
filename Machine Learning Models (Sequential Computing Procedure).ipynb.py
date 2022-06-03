# Databricks notebook source
# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import collections
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('words')
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import string
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining of Clean Text class

# COMMAND ----------

class CleanText(BaseEstimator, TransformerMixin):
  # Remove any usage of @
  def remove_mentions(self, input_text):
    return re.sub(r'@\w+', '', input_text)
  
  # Remove any usage urls
  def remove_urls(self, input_text):
    return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
  
  # Compressing of underscore and the emoji to be kept as one word
  def emoji_oneword(self, input_text):
    return input_text.replace('_','')
    
  # Removing punctuations from the tweets
  def remove_punctuation(self, input_text):
    # Make translation table
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
    return input_text.translate(trantab)
  
  # Removing bumbers from the tweets
  def remove_digits(self, input_text):
    return re.sub('\d+', '', input_text)
  
  # Converting all characters into lower case
  def to_lower(self, input_text):
    return input_text.lower()
    
  # Removing all stopwords as they hold no values to the resulting sentiment analysis
  def remove_stopwords(self, input_text):
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words)
    
  # Normalise words with potentially same definition  
  def stemming(self, input_text):
    porter = PorterStemmer()
    words = input_text.split() 
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)
    
  # Fitting
  def fit(self, X, y=None, **fit_params):
    return self
    
  # Transforming of tweets using all the declared functions within this class
  def transform(self, X, **transform_params):
    clean_X = X.apply(self.remove_mentions).\
    apply(self.remove_urls).\
    apply(self.emoji_oneword).\
    apply(self.remove_punctuation).\
    apply(self.remove_digits).\
    apply(self.to_lower).\
    apply(self.remove_stopwords).\
    apply(self.stemming)
    return clean_X  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing of Biden's dataset as dataframe

# COMMAND ----------

# 17500 data instances from each dataset is the maximum processing capacity in databrick
df_biden = spark.read.parquet("/mnt/noted/tweets_biden_pyspark_processed.parquet").limit(17500)
# Use line below if you want to read file locally
# df_biden = spark.read.parquet("tweets_biden_pyspark_processed.parquet").limit(17500)
df_biden = df_biden.toPandas()
# Converting negative sentiments in Biden dataset into 1 and positive sentiments into 2
df_biden['Sentiment_Overall'].replace({-1: 1, 1: 2}, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing of Trump's Dataset as dataframe

# COMMAND ----------

# 17500 data instances from each dataset is the maximum processing capacity in databrick
df_trump = spark.read.parquet("/mnt/noted/tweets_trump_pyspark_processed.parquet").limit(17500)
# Use line below if you want to read file locally
# df_trump = spark.read.parquet("/mnt/noted/tweets_trump_pyspark_processed.parquet").limit(17500)
df_trump = df_trump.toPandas()
# Converting negative sentiments in Trump dataset into 3 and positive sentiments into 4
df_trump['Sentiment_Overall'].replace({-1: 3, 1: 4}, inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merging of Biden's and Trump's dataframe into one

# COMMAND ----------

# Concatenating of the imported datasets into one
df_merge = pd.concat([df_biden, df_trump])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application of text cleaning

# COMMAND ----------

# Application of text cleaning using CleanText class
ct = CleanText()
sr_clean = ct.fit_transform(df_merge.tweet)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting sentences into a vectorised format

# COMMAND ----------

# Converting word within tweets into a vectorised format to be fed into the training of machine learning technique
cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])
fig, ax = plt.subplots(figsize=(12,10))
sns.barplot(x="word", y = 'freq', data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialise Random Forest

# COMMAND ----------

RandomForest = RandomForestClassifier(n_estimators = 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialise Naive Bayes

# COMMAND ----------

gnb = GaussianNB()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Test Splitting

# COMMAND ----------

# Performing a 80% 20% training and testing dataset split
X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame.sparse.from_spmatrix(bow).to_numpy(),\
                                                    df_merge['Sentiment_Overall'].to_frame().to_numpy().ravel(), \
                                                    test_size = 0.2,\
                                                    random_state = 42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Accuracy

# COMMAND ----------

# Fitting of Random Forest classifier
RandomForest.fit(X_train, Y_train)
predictions_rf = RandomForest.predict(X_test)
rf_score = accuracy_score(Y_test, predictions_rf)
print('Random Forest Accuracy: %.3f' % rf_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Naive Bayes Accuracy

# COMMAND ----------

# Fitting of Naive Bayes classifier
gnb.fit(X_train, Y_train)
predictions_gnb = gnb.predict(X_test)
gnb_score = accuracy_score(Y_test, predictions_gnb)
print('Naive Bayes Accuracy: %.3f' % gnb_score)