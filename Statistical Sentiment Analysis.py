# Databricks notebook source
# import necessary libraries

from pyspark.sql import functions as F
import pandas as pd
import nltk
from pyspark.sql.types import IntegerType
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

# COMMAND ----------

#read data in parquet and csv format

dt_tweet = spark.read.parquet("/mnt/noted/tweets_trump_pyspark_processed.parquet")
biden_tweet = spark.read.parquet("/mnt/noted/tweets_biden_pyspark_processed.parquet")
actual_election_result = spark.read.format('csv').option("header", 'true').option('multiLine','true').load("/mnt/noted/ActualUSElectionResult2020.csv")
# Use line below if you want to read files locally
# dt_tweet = spark.read.parquet("tweets_trump_pyspark_processed.parquet")
# biden_tweet = spark.read.parquet("tweets_biden_pyspark_processed.parquet")
# actual_election_result = spark.read.format('csv').option("header", 'true').option('multiLine','true').load("ActualUSElectionResult2020.csv")

# COMMAND ----------

# defining country to be included for analaysis
country = ['United States','United States of America']

# defining states to be excluded for analaysis
excluded_state = ['Guam', 'Puerto Rico']

# COMMAND ----------

#filtering valid data entry for dataset that includes tweets related to Donald Trump

dt_us_tweet = dt_tweet.filter(F.col('country').isin(country)).filter(F.col('state').isin(excluded_state) == False).na.drop(subset=["state"]).withColumn('timestamp',F.to_date(F.regexp_replace(F.col("user_join_date"), "/", "-"), 'MM-dd-yyyy'))
display(dt_us_tweet)

# COMMAND ----------

#filtering valid data entry for dataset that includes tweets related to Joe Biden

biden_us_tweet = biden_tweet.filter(F.col('country').isin(country)).filter(F.col('state').isin(excluded_state) == False).na.drop(subset=["state"]).withColumn('timestamp',F.to_date(F.regexp_replace(F.col("user_join_date"), "/", "-"), 'MM-dd-yyyy'))
display(biden_us_tweet)

# COMMAND ----------

# Generate histogram diagram based on the sentiment polarity of collected tweets

dt_us_tweet_sentiment = dt_us_tweet.groupby('Sentiment_Overall').count().withColumnRenamed("count","Donald Trump")
biden_us_tweet_sentiment = biden_us_tweet.groupby('Sentiment_Overall').count().withColumnRenamed("count","Joe Biden")
compound_us_tweet_sentiment = dt_us_tweet_sentiment.join(biden_us_tweet_sentiment,on=['Sentiment_Overall'])
display(compound_us_tweet_sentiment)
# positive:1 ; negative:-1 ; neutral: 2

# COMMAND ----------

# Generate histogram diagram based on the states in US using only positive sentiment of collected tweets

dt_compound_state_tweet_sentiment = dt_us_tweet.filter(F.col('Sentiment_Overall')==1).groupby('state').agg(F.sum('Sentiment_Overall').alias('Donald Trump'))
biden_compound_state_tweet_sentiment = biden_us_tweet.filter(F.col('Sentiment_Overall')==1).groupby('state').agg(F.sum('Sentiment_Overall').alias('Joe Biden'))
compound_state_us_tweet_sentiment = dt_compound_state_tweet_sentiment.join(biden_compound_state_tweet_sentiment,on=['state'])
display(compound_state_us_tweet_sentiment.take(10))

# COMMAND ----------

# Deciding the nominees with more supporter based on the number of their positive tweets in each states

compound_state_us_tweet_sentiment_with_flag = compound_state_us_tweet_sentiment.withColumn('More Supporter', F.when((F.col("Donald Trump") < F.col("Joe Biden")), 'Joe Biden').otherwise('Donald Trump'))
display(compound_state_us_tweet_sentiment_with_flag)

# COMMAND ----------

# Defining the electoral votes in each states along side with the comparison done previously 

compound_state_us_tweet_sentiment_with_electoral_votes = compound_state_us_tweet_sentiment_with_flag.join(actual_election_result.select("States","Electoral_Votes").withColumnRenamed("States","state"), on=['state'])
display(compound_state_us_tweet_sentiment_with_electoral_votes)

# COMMAND ----------

# Calculating the electoral votes of each nominees

display(compound_state_us_tweet_sentiment_with_electoral_votes.withColumn("Electoral_Votes",compound_state_us_tweet_sentiment_with_electoral_votes["Electoral_Votes"].cast(IntegerType())).groupBy("More Supporter").agg(F.sum('Electoral_Votes').alias('Electoral Vote')))

# COMMAND ----------

# MAGIC %md
# MAGIC SA_USER

# COMMAND ----------

# Generate histogram diagram based on the sentiment polarity of user in collected tweets

dt_us_user = dt_us_tweet.dropDuplicates(["user_id"])
dt_us_user_sentiment = dt_us_user.groupby('Sentiment_Overall').count().withColumnRenamed("count","Donald Trump")

biden_us_user = biden_us_tweet.dropDuplicates(["user_id"])
biden_us_user_sentiment = biden_us_user.groupby('Sentiment_Overall').count().withColumnRenamed("count","Joe Biden")

compound_us_user_sentiment = dt_us_user_sentiment.join(biden_us_user_sentiment,on=['Sentiment_Overall'])
display(compound_us_user_sentiment)
# positive:1 ; negative:-1 ; neutral: 2

# COMMAND ----------

# Generate histogram diagram based on the states in US using only positive sentiment of user in the dataset

dt_compound_state_tweet_user_sentiment = dt_us_user.filter(F.col('Sentiment_Overall')==1).groupby('state').agg(F.sum('Sentiment_Overall').alias('Donald Trump'))
biden_compound_state_tweet_user_sentiment = biden_us_user.filter(F.col('Sentiment_Overall')==1).groupby('state').agg(F.sum('Sentiment_Overall').alias('Joe Biden'))
compound_state_us_tweet_user_sentiment = dt_compound_state_tweet_user_sentiment.join(biden_compound_state_tweet_user_sentiment,on=['state'])
display(compound_state_us_tweet_user_sentiment.take(10))

# COMMAND ----------

# Deciding the nominees with more supporter based on the number of the users with positve sentiment towards respective nominees in each states

compound_state_us_tweet_user_sentiment_with_flag = compound_state_us_tweet_user_sentiment.withColumn('More Supporter', F.when((F.col("Donald Trump") < F.col("Joe Biden")), 'Joe Biden').otherwise('Donald Trump'))
display(compound_state_us_tweet_user_sentiment_with_flag)

# COMMAND ----------

# Defining the electoral votes in each states along side with the comparison done previously 

compound_state_us_tweet_user_sentiment_with_electoral_votes = compound_state_us_tweet_user_sentiment_with_flag.join(actual_election_result.select("States","Electoral_Votes").withColumnRenamed("States","state"), on=['state'])
display(compound_state_us_tweet_user_sentiment_with_electoral_votes)

# COMMAND ----------

# Calculating the electoral votes of each nominees

display(compound_state_us_tweet_user_sentiment_with_electoral_votes.withColumn("Electoral_Votes",compound_state_us_tweet_sentiment_with_electoral_votes["Electoral_Votes"].cast(IntegerType())).groupBy("More Supporter").agg(F.sum('Electoral_Votes').alias('Electoral Vote')))

# COMMAND ----------

# MAGIC %md
# MAGIC With fake account removed

# COMMAND ----------

# Generate histogram diagram based on the sentiment polarity of actual user in collected tweets

dt_us_real_user = dt_us_user.filter(F.col("timestamp")<(F.lit("2020-09-01"))) 
dt_us_real_user_sentiment = dt_us_user.groupby('Sentiment_Overall').count().withColumnRenamed("count","Donald Trump")

biden_us_real_user = biden_us_user.filter(F.col("timestamp")<(F.lit("2020-09-01"))) 
biden_us_real_user_sentiment = biden_us_user.groupby('Sentiment_Overall').count().withColumnRenamed("count","Joe Biden")

compound_us_real_user_sentiment = dt_us_real_user_sentiment.join(biden_us_real_user_sentiment,on=['Sentiment_Overall'])
display(compound_us_real_user_sentiment)
# positive:1 ; negative:-1 ; neutral: 2

# COMMAND ----------

# Generate histogram diagram based on the states in US using only positive sentiment of actual user in the dataset

dt_compound_state_tweet_real_user_sentiment = dt_us_real_user.filter(F.col('Sentiment_Overall')==1).groupby('state').agg(F.sum('Sentiment_Overall').alias('Donald Trump'))
biden_compound_state_tweet_real_user_sentiment = biden_us_user.filter(F.col('Sentiment_Overall')==1).groupby('state').agg(F.sum('Sentiment_Overall').alias('Joe Biden'))
compound_state_us_tweet_real_user_sentiment = dt_compound_state_tweet_real_user_sentiment.join(biden_compound_state_tweet_real_user_sentiment,on=['state'])
display(compound_state_us_tweet_real_user_sentiment.take(10))

# COMMAND ----------

# Deciding the nominees with more supporter based on the number of the actual users with positve sentiment towards respective nominees in each states

compound_state_us_tweet_real_user_sentiment_with_flag = compound_state_us_tweet_real_user_sentiment.withColumn('More Supporter', F.when((F.col("Donald Trump") < F.col("Joe Biden")), 'Joe Biden').otherwise('Donald Trump'))
display(compound_state_us_tweet_real_user_sentiment_with_flag)

# COMMAND ----------

# Defining the electoral votes in each states along side with the comparison done previously 

compound_state_us_tweet_real_user_sentiment_with_electoral_votes = compound_state_us_tweet_real_user_sentiment_with_flag.join(actual_election_result.select("States","Electoral_Votes").withColumnRenamed("States","state"), on=['state'])
display(compound_state_us_tweet_real_user_sentiment_with_electoral_votes)

# COMMAND ----------

# Calculating the electoral votes of each nominees

display(compound_state_us_tweet_real_user_sentiment_with_electoral_votes.withColumn("Electoral_Votes",compound_state_us_tweet_sentiment_with_electoral_votes["Electoral_Votes"].cast(IntegerType())).groupBy("More Supporter").agg(F.sum('Electoral_Votes').alias('Electoral Vote')))

# COMMAND ----------

