# Databricks notebook source
# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark import SparkContext
from sklearn.base import BaseEstimator, TransformerMixin
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.functions import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import re
import string
import nltk
import pandas as pd
nltk.download('stopwords')
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import time


# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining of Clean Text class

# COMMAND ----------

class CleanText(BaseEstimator, TransformerMixin):
  
  def __init__(self, stopwords):
    self.stopword = stopwords
  
  def remove_mentions(self, input_text):
    return re.sub(r'@\w+', '', input_text)
  
  def remove_urls(self, input_text):
    return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
  
  def emoji_oneword(self, input_text):
    # By compressing the underscore, the emoji is kept as one word
    return input_text.replace('_','')
    
  def remove_punctuation(self, input_text):
    # Make translation table
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
    return input_text.translate(trantab)
    
  def remove_digits(self, input_text):
    return re.sub('\d+', '', input_text)
    
  def to_lower(self, input_text):
    return input_text.lower()
    
  def remove_stopwords(self, input_text):
    stopwords_list = self.stopword
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 
    
  def stemming(self, input_text):
    porter = PorterStemmer()
    words = input_text.split() 
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)
    
  def fit(self, X, y=None, **fit_params):
    return self
    
  def transform(self, X, **transform_params):
    clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
    return clean_X

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Biden dataset

# COMMAND ----------

df_biden = spark.read.parquet("/mnt/noted/tweets_biden_pyspark_processed.parquet").repartition(32)
# Use this line if you want to read locally
# df_biden = spark.read.parquet("tweets_biden_pyspark_processed.parquet").repartition(32)

df_biden = df_biden.withColumn('Sentiment_Overall', when(df_biden.Sentiment_Overall == '-1', regexp_replace('Sentiment_Overall', '-1', '1')).when(df_biden.Sentiment_Overall == '1', regexp_replace('Sentiment_Overall', '1', '2')).when(df_biden.Sentiment_Overall == '0', regexp_replace('Sentiment_Overall', '0', '0')))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Trump dataset

# COMMAND ----------

df_trump = spark.read.parquet("/mnt/noted/tweets_trump_pyspark_processed.parquet").repartition(32)
# Use this line if you want to read file locally
# df_trump = spark.read.parquet("tweets_trump_pyspark_processed.parquet").repartition(32)

df_trump = df_trump.withColumn('Sentiment_Overall', when(df_trump.Sentiment_Overall == '-1', regexp_replace('Sentiment_Overall', '-1', '3')).when(df_trump.Sentiment_Overall == '1', regexp_replace('Sentiment_Overall', '1', '4')).when(df_trump.Sentiment_Overall == '0', regexp_replace('Sentiment_Overall', '0', '0')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge filtered Biden and Trump Dataset

# COMMAND ----------

df = df_biden.unionByName(df_trump)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Cleaning

# COMMAND ----------

stopwords_list = stopwords.words('english')
ct = CleanText(stopwords_list)

udf_func = udf(lambda x: (ct.fit_transform(pd.Series(data=x))[0]))
df = df.withColumn("srclean", udf_func(col("tweet")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tokenise Sentences

# COMMAND ----------

tokenizer = Tokenizer(inputCol="srclean", outputCol="words")
df = tokenizer.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vectorise Sentences

# COMMAND ----------

cv = CountVectorizer(inputCol="words", outputCol="vectorised")

vectorise = cv.fit(df)
df = vectorise.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting "Sentiment_Overall" data type to integer

# COMMAND ----------

df = df.withColumn("Sentiment_Overall", df["Sentiment_Overall"].cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Traning and testing data split

# COMMAND ----------

train, test = df.randomSplit([0.8,0.2])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialise Random Forest

# COMMAND ----------

rf = RandomForestClassifier(featuresCol="vectorised", labelCol="Sentiment_Overall", numTrees=200, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialise Naive Bayes

# COMMAND ----------

nb = NaiveBayes(featuresCol='vectorised', labelCol='Sentiment_Overall', modelType="multinomial")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialise evaluation function

# COMMAND ----------

eval = MulticlassClassificationEvaluator(labelCol="Sentiment_Overall", predictionCol="prediction", metricName='accuracy')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Random Forest model

# COMMAND ----------

rfmodel = rf.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Naive Bayes model

# COMMAND ----------

nbmodel = nb.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing and evaluating trained Random Forest model on test data

# COMMAND ----------

pred_rf = rfmodel.transform(test)
acc_rf = eval.evaluate(pred_rf)
print("accuracy for Random Forest = %g" % acc_rf)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Testing and evaluating trained Naive Bayes model on test data

# COMMAND ----------

pred_nb = nbmodel.transform(test)
acc_nb = eval.evaluate(pred_nb)
print("Accuracy for Naive Bayes = %g" % acc_nb)