# Databricks notebook source
# MAGIC %pip install --upgrade pip
# MAGIC %pip install langdetect

# COMMAND ----------

# Import packages needed for the experiment
import pandas as pd
import pyspark.pandas as ps
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
# Download vader model from nltk library
nltk.download('vader_lexicon')
# Assign vader sentiment analyzer
sid = SentimentIntensityAnalyzer()

# COMMAND ----------

# Read data from #Trump and convert into pandas dataframe
tweets_trump_df = spark.read.format('csv').option("header", 'true').option('multiLine','true').load("/mnt/noted/dt_tweets.csv").limit(100000)
# Use this line if you want to read locally
# tweets_trump_df = spark.read.format('csv').option("header", 'true').option('multiLine','true').load("/mnt/noted/dt_tweets.csv").limit(100000)
tweets_trump = tweets_trump_df.toPandas()
tweets_trump.drop(tweets_trump.columns.difference(['user_join_date','tweet','user_id','city','country','state','state_code']), axis = 1, inplace = True)
tweets_trump

# COMMAND ----------

# Read data from #Biden and convert into pandas dataframe
tweets_biden_df = spark.read.format('csv').option("header", 'true').option('multiLine','true').load("/mnt/noted/jb_tweets.csv").limit(100000)
tweets_biden = tweets_biden_df.toPandas()
tweets_biden.drop(tweets_biden.columns.difference(['user_join_date','tweet','user_id','city','country','state','state_code']), axis = 1, inplace = True)
tweets_biden

# COMMAND ----------

import re
# Pre-processing of datas
# Remove instances that has biden mentions in #Trump
# Remove instances that has trump mentions in #Biden
# Remove hyperlinks, @, \n, and "RT" reply tag on tweets
# Tweets that are not in english detected using langdetect library are returned as null value
def clean_tweets(text,mention):
    
    if mention in text.lower():
      return ''
    
    text = re.sub("RT @[\w]*:","",str(text))
    text = re.sub("@[\w]*","",str(text))
    text = re.sub("https?://[A-Za-z0-9./]*","",str(text))
    text = re.sub("\n","",str(text))    
    
    try:
      if detect(text) != 'en':
        return ''
      else:
        return text
    except:
      return ''
    
    return text
  
# Drop instances that has null in user_id column
tweets_trump = tweets_trump.dropna(subset = ['user_id'])
tweets_biden = tweets_biden.dropna(subset = ['user_id'])

# Apply clean tweet function into each instances
tweets_trump['tweet'] = tweets_trump['tweet'].apply(lambda x: clean_tweets(x,'biden'))
tweets_biden['tweet'] = tweets_biden['tweet'].apply(lambda x: clean_tweets(x,'trump'))

# Remove instances that has null value in tweet column
tweets_trump = tweets_trump.loc[tweets_trump['tweet'] != '']
tweets_biden = tweets_biden.loc[tweets_biden['tweet'] != '']


# Display table
tweets_trump
tweets_biden

# COMMAND ----------

# Create empty column for Sentiment and Sentiment Overall for both trump and biden table
tweets_trump["Sentiment"] = np.nan
tweets_biden["Sentiment"] = np.nan
tweets_trump["Sentiment_Overall"] = np.nan
tweets_biden["Sentiment_Overall"] = np.nan

# Apply Sentiment Intensity Analyzer function to each of the tweets
tweets_trump["Sentiment"] = tweets_trump['tweet'].apply(lambda x: sid.polarity_scores(x))
tweets_biden["Sentiment"] = tweets_biden['tweet'].apply(lambda x: sid.polarity_scores(x))

# Calculate Sentiment Overall using compound value in Sentiment column
def sentimentVerdict(sentiment):
    # Compound positive value means tweet has positive sentiment toward that candidate
    if sentiment['compound'] > 0:
        return 1
    # Compound negative value means tweet has negative sentiment toward that candidate  
    elif sentiment['compound'] < -0:
        return 0
    # If 0 means tweet is neutral toward both candidate
    else:
        return 2

# Apply sentimentVerdict which calculates the Sentiment Overall of each of the tweets
tweets_trump['Sentiment_Overall'] = tweets_trump['Sentiment'].apply(lambda x: sentimentVerdict(x))
tweets_biden['Sentiment_Overall'] = tweets_biden['Sentiment'].apply(lambda x: sentimentVerdict(x))
tweets_trump

# COMMAND ----------

# Drop instances that has neutral sentiment overall
tweets_trump = tweets_trump.loc[tweets_trump['Sentiment_Overall'] != 2]
tweets_biden = tweets_biden.loc[tweets_biden['Sentiment_Overall'] != 2]

# Drop sentiment column
tweets_trump.drop(columns = ['Sentiment'])
tweets_biden.drop(columns = ['Sentiment'])


tweets_trump
tweets_biden

# COMMAND ----------

# Shows all the election states located in the US
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New York', 'New Mexico', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
stateCodes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

# Map short form state names into Full names
stateMapping = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
                  'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NY': 'New York', 'NM': 'New Mexico', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT':  'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV':  'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}

# Map Full name of states into short form
stateMappingInverse = {'District of Columbia': 'DC','Alabama': 'AL','Montana': 'MT','Alaska': 'AK','Nebraska': 'NE','Arizona': 'AZ','Nevada': 'NV','Arkansas': 'AR','New Hampshire': 'NH','California': 'CA','New Jersey': 'NJ','Colorado': 'CO','New Mexico': 'NM','Connecticut': 'CT','New York': 'NY','Delaware': 'DE','North Carolina': 'NC','Florida': 'FL','North Dakota': 'ND','Georgia': 'GA','Ohio': 'OH','Hawaii': 'HI','Oklahoma': 'OK','Idaho': 'ID','Oregon': 'OR','Illinois': 'IL','Pennsylvania': 'PA','Indiana': 'IN','Rhode Island': 'RI','Iowa': 'IA','South Carolina': 'SC','Kansas': 'KS','South Dakota': 'SD','Kentucky': 'KY','Tennessee': 'TN','Louisiana': 'LA','Texas': 'TX','Maine': 'ME','Utah': 'UT','Maryland': 'MD','Vermont': 'VT','Massachusetts': 'MA','Virginia': 'VA','Michigan': 'MI','Washington': 'WA','Minnesota': 'MN','West Virginia': 'WV','Mississippi': 'MS','Wisconsin': 'WI','Missouri': 'MO', 'Wyoming': 'WY',
}

# Filter out all tweets that are not tweeted in the US and not supporting the particular candidate
tweets_trump = tweets_trump[((tweets_trump['country'] == 'United States') | (tweets_trump['country'] == 'United States of America')) & (tweets_trump['state'] != 'Guam') & (tweets_trump['state'] != 'Puerto Rico') & (tweets_trump['Sentiment_Overall'] == 1)]
tweets_biden = tweets_biden[((tweets_biden['country'] == 'United States') | (tweets_biden['country'] == 'United States of America')) & (tweets_biden['state'] != 'Guam') & (tweets_biden['state'] != 'Puerto Rico') & (tweets_biden['Sentiment_Overall'] == 1)]

# Merge both tweets from #trump and tweets from #biden
df = pd.merge(tweets_trump['state'].value_counts(), tweets_biden['state'].value_counts(), right_index = True, 
               left_index = True)
df = df.rename(columns = {"state_x": "Total Trump Mentions", "state_y": "Total Biden Mentions"})

# Plot the supports of both candidates located in each of US states
ax = df.plot(kind='barh', figsize=(16, 25), zorder=2)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#Replacing ticks with horizontal lines
#ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
vals = ax.get_xticks()
for tick in vals:
      ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set y-axis label
ax.set_ylabel("States", labelpad=20, weight='bold', size=12)
ax.set_title('Comparison of Twitter supports of both candidates in all US states as per data collected',fontweight="bold", size=15)

# COMMAND ----------

# Group data by state with sentiment overall counts
tweets_trump_location = tweets_trump.groupby(['state', 'Sentiment_Overall']).count()
tweets_trump_location = tweets_trump_location['user_id']
tweets_biden_location = tweets_biden.groupby(['state', 'Sentiment_Overall']).count()
tweets_biden_location = tweets_biden_location['user_id']

# Create a new dataframe that includes the tweet supports and candidate who won for each candidates in each states
tweets_location_df = pd.DataFrame({'State': [state for state in states],
        'State Code': [stateMappingInverse[state] for state in states],
        'Trump Positive': [0 for state in states],
        'Biden Positive': [0 for state in states],
        'Trump/Biden Ratio':[0.00000000 for state in states],
        'Who Won': ['' for state in states]})
  
tweets_location_df.set_index('State', inplace = True)
for state in states:
  positiveTrump,  positiveBiden = 0,0
  try:
    positiveTrump = tweets_trump_location[state][1]
  except:
    positiveTrump = 0
  
  try:
    positiveBiden = tweets_biden_location[state][1]
  except:
    positiveBiden = 0
    

  if positiveTrump == 0:
    tweets_location_df.at[state, 'Trump Positive'] = 0
  else:
    tweets_location_df.at[state, 'Trump Positive'] = positiveTrump
    
  if positiveBiden == 0:
    tweets_location_df.at[state, 'Biden Positive'] = 0
  else:
    tweets_location_df.at[state, 'Biden Positive'] = positiveBiden
  
  # Divide Trump supports with Biden supports if more than 1 Trump wins the state otherwise Biden
  tweets_location_df.at[state,'Trump/Biden Ratio'] = positiveTrump/positiveBiden
  if positiveTrump - positiveBiden > 0:
    tweets_location_df.at[state,'Who Won'] = 'Trump'
  else:
    tweets_location_df.at[state,'Who Won'] = 'Biden'
  
tweets_location_df

# COMMAND ----------

import plotly.express as px

# Plots a map that shows the ratio of Trump supports / Biden supports in each state 
fig = px.choropleth(tweets_location_df,
                    locations='State Code',
                    locationmode="USA-states",
                    scope='usa',
                    color='Trump/Biden Ratio',
                    color_continuous_scale=('#4040ff', '#ff4040'),
                    range_color=(0,2),
                    color_continuous_midpoint=0,
                    )
fig.show()

# COMMAND ----------

# Reads the actual election results from a csv
actual_election_result = spark.read.format('csv').option("header", 'true').option('multiLine','true').load("/mnt/noted/ActualUSElectionResult2020.csv")
actual_election_resultPandas = actual_election_result.toPandas()
actual_election_resultPandasInplace = actual_election_resultPandas.rename(columns = {'States' : 'State'})
actual_election_resultPandasInplace.set_index('State', inplace = True)
actual_election_resultPandasInplace

# COMMAND ----------

# Calculate the amount of similar result between tweet supports and actual election
sameElectionResult = 0

for state in states:
  if tweets_location_df.at[state, 'Who Won'] == actual_election_resultPandasInplace.at[state, 'Who_Win']:
    sameElectionResult = sameElectionResult + 1

print("Total number of same Election Result of every state")
sameElectionResult

# COMMAND ----------

# Calculates the amount of electoral votes won for both candidates in actual election
actual_election_resultPandas = pd.DataFrame(actual_election_resultPandas, columns = ['Who_Win','Electoral_Votes'])
actual_election_resultPandas['Electoral_Votes'] = pd.to_numeric(actual_election_resultPandas['Electoral_Votes'])
actualElectoralVoteResult = actual_election_resultPandas.groupby(['Who_Win', 'Electoral_Votes']).count()

bidenElectoralVotes = actual_election_resultPandas.loc[actual_election_resultPandas['Who_Win'] == 'Biden', 'Electoral_Votes'].sum()
bidenElectoralVotes
trumpElectoralVotes = actual_election_resultPandas.loc[actual_election_resultPandas['Who_Win'] == 'Trump', 'Electoral_Votes'].sum()
trumpElectoralVotes

actualResult = pd.DataFrame({'Team':['Biden','Trump'],
                             'Electoral Votes': [bidenElectoralVotes,trumpElectoralVotes]})
print("Actual Election Result")
actualResult

# COMMAND ----------

# Calculates the amount of electoral votes won for both candidates in tweet supports
twitterResult = tweets_location_df.join(actual_election_resultPandasInplace,['State'],how='inner')
twitterResult = twitterResult[['Who Won','Electoral_Votes']]
twitterResult['Electoral_Votes'] = pd.to_numeric(twitterResult['Electoral_Votes'])

bidenElectoralVotes = twitterResult.loc[twitterResult['Who Won'] == 'Biden', 'Electoral_Votes'].sum()
bidenElectoralVotes
trumpElectoralVotes = twitterResult.loc[twitterResult['Who Won'] == 'Trump', 'Electoral_Votes'].sum()

twitterResultEnd = pd.DataFrame({'Team':['Biden','Trump'],
                             'Electoral Votes': [bidenElectoralVotes,trumpElectoralVotes]})
print("Twitter Election Result")
twitterResultEnd

# COMMAND ----------



# COMMAND ----------

## Filter duplicated user id in user_id column and keep the last tweet from every unique user id
tweets_trump = tweets_trump.drop_duplicates(subset=['user_id'], keep='last')
tweets_biden = tweets_biden.drop_duplicates(subset=['user_id'], keep='last')
# Convert user_join_date column datatype to pandas datetime
tweets_trump['user_join_date'] = pd.to_datetime(tweets_trump['user_join_date'])
tweets_biden['user_join_date'] = pd.to_datetime(tweets_biden['user_join_date'])
# Filter out tweets that made by user who joined after 1/9/2022
tweets_trump = tweets_trump[(tweets_trump['user_join_date'].dt.year <= 2022) | (tweets_trump['user_join_date'].dt.month <= 9) | (tweets_trump['user_join_date'].dt.day <= 1)]
tweets_biden = tweets_biden[(tweets_biden['user_join_date'].dt.year <= 2022) | (tweets_biden['user_join_date'].dt.month <= 9) | (tweets_biden['user_join_date'].dt.day <= 1)]
# Merge both hashtag's table
df = pd.merge(tweets_trump['state'].value_counts(), tweets_biden['state'].value_counts(), right_index = True, 
               left_index = True)
df = df.rename(columns = {"state_x": "Total Trump Mentions", "state_y": "Total Biden Mentions"})

tweets_trump
# Plot the supports of both candidates located in each of US states
ax = df.plot(kind='barh', figsize=(16, 25), zorder=2)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#Replacing ticks with horizontal lines
#ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
vals = ax.get_xticks()
for tick in vals:
      ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set y-axis label
ax.set_ylabel("States", labelpad=20, weight='bold', size=12)
ax.set_title('Comparison of Twitter users of both candidates in all US states as per data collected',fontweight="bold", size=15)

# COMMAND ----------

# Group data by state with sentiment overall counts
tweets_trump_location = tweets_trump.groupby(['state', 'Sentiment_Overall']).count()
tweets_trump_location = tweets_trump_location['user_id']
tweets_biden_location = tweets_biden.groupby(['state', 'Sentiment_Overall']).count()
tweets_biden_location = tweets_biden_location['user_id']

# Create a new dataframe that includes the tweet supports and candidate who won for each candidates in each states
tweets_location_df = pd.DataFrame({'State': [state for state in states],
        'State Code': [stateMappingInverse[state] for state in states],
        'Trump Positive': [0 for state in states],
        'Biden Positive': [0 for state in states],
        'Trump/Biden Ratio':[0.00000000 for state in states],
        'Who Won': ['' for state in states]})
  
tweets_location_df.set_index('State', inplace = True)
for state in states:
  positiveTrump,  positiveBiden = 0,0
  try:
    positiveTrump = tweets_trump_location[state][1]
  except:
    positiveTrump = 0
  
  try:
    positiveBiden = tweets_biden_location[state][1]
  except:
    positiveBiden = 0
    

  if positiveTrump == 0:
    tweets_location_df.at[state, 'Trump Positive'] = 0
  else:
    tweets_location_df.at[state, 'Trump Positive'] = positiveTrump
    
  if positiveBiden == 0:
    tweets_location_df.at[state, 'Biden Positive'] = 0
  else:
    tweets_location_df.at[state, 'Biden Positive'] = positiveBiden
  
  # Divide Trump supports with Biden supports if more than 1 Trump wins the state otherwise Biden
  tweets_location_df.at[state,'Trump/Biden Ratio'] = positiveTrump/positiveBiden
  if positiveTrump - positiveBiden > 0:
    tweets_location_df.at[state,'Who Won'] = 'Trump'
  else:
    tweets_location_df.at[state,'Who Won'] = 'Biden'
  
tweets_location_df.display()

# COMMAND ----------

import plotly.express as px
# Plots a map that shows the ratio between Trump supports / Biden supports in twitter users
fig = px.choropleth(tweets_location_df,
                    locations='State Code',
                    locationmode="USA-states",
                    scope='usa',
                    color='Trump/Biden Ratio',
                    color_continuous_scale=('#4040ff', '#ff4040'),
                    range_color=(0,2),
                    color_continuous_midpoint=0,
                    )
fig.show()

# COMMAND ----------

# Calculate the amount of similar result between tweet supports and actual election
sameElectionResult = 0

for state in states:
  if tweets_location_df.at[state, 'Who Won'] == actual_election_resultPandasInplace.at[state, 'Who_Win']:
    sameElectionResult = sameElectionResult + 1

print("Total number of same Election Result of every state")
sameElectionResult

# COMMAND ----------

# Calculates the amount of electoral votes won for both candidates in actual election
actual_election_resultPandas = pd.DataFrame(actual_election_resultPandas, columns = ['Who_Win','Electoral_Votes'])
actual_election_resultPandas['Electoral_Votes'] = pd.to_numeric(actual_election_resultPandas['Electoral_Votes'])
actualElectoralVoteResult = actual_election_resultPandas.groupby(['Who_Win', 'Electoral_Votes']).count()

bidenElectoralVotes = actual_election_resultPandas.loc[actual_election_resultPandas['Who_Win'] == 'Biden', 'Electoral_Votes'].sum()
bidenElectoralVotes
trumpElectoralVotes = actual_election_resultPandas.loc[actual_election_resultPandas['Who_Win'] == 'Trump', 'Electoral_Votes'].sum()
trumpElectoralVotes

actualResult = pd.DataFrame({'Team':['Biden','Trump'],
                             'Electoral Votes': [bidenElectoralVotes,trumpElectoralVotes]})
print("Actual Election Result")
actualResult

# COMMAND ----------

# Calculates the amount of electoral votes won for both candidates in tweet users
twitterResult = tweets_location_df.join(actual_election_resultPandasInplace,['State'],how='inner')
twitterResult = twitterResult[['Who Won','Electoral_Votes']]
twitterResult['Electoral_Votes'] = pd.to_numeric(twitterResult['Electoral_Votes'])

bidenElectoralVotes = twitterResult.loc[twitterResult['Who Won'] == 'Biden', 'Electoral_Votes'].sum()
bidenElectoralVotes
trumpElectoralVotes = twitterResult.loc[twitterResult['Who Won'] == 'Trump', 'Electoral_Votes'].sum()

twitterResultEnd = pd.DataFrame({'Team':['Biden','Trump'],
                             'Electoral Votes': [bidenElectoralVotes,trumpElectoralVotes]})
print("Twitter Election Result")
twitterResultEnd

# COMMAND ----------



# COMMAND ----------

# Convert Biden and Trump column to pandas numeric
actual_election_resultPandasInplace['Biden'] = pd.to_numeric(actual_election_resultPandasInplace['Biden'])
actual_election_resultPandasInplace['Trump'] = pd.to_numeric(actual_election_resultPandasInplace['Trump'])
# Calculates Trump/Biden Ratio for each states
actual_election_resultPandasInplace['Trump/Biden Ratio'] = actual_election_resultPandasInplace['Trump']/actual_election_resultPandasInplace['Biden']
actual_election_resultPandasInplace = actual_election_resultPandasInplace.loc[:,~actual_election_resultPandasInplace.columns.duplicated()]
actual_election_resultPandasInplace

# COMMAND ----------

tweets_location_df = tweets_location_df.rename(columns = {'Trump/Biden Ratio':'Trump/Biden Ratio Tweets'})
actual_election_resultPandasInplace = actual_election_resultPandasInplace.rename(columns = {'Trump/Biden Ratio':'Trump/Biden Ratio Actual'})

# Creats a table where it shows Trump/Biden Ratio in actual election and Trump/Biden Ratio in Tweet users
compareRatioCorr = tweets_location_df.join(actual_election_resultPandasInplace,['State'],how = 'inner')
compareRatioCorr = pd.DataFrame(compareRatioCorr,columns=['Trump/Biden Ratio Actual','Trump/Biden Ratio Tweets'])
compareRatioCorr

# COMMAND ----------

# Calculates Pearson Correlation Coefficient between Trump/Biden Ratio in actual election
# and Trump/Biden Ratio in Tweet users
corr = compareRatioCorr.corr(method='pearson')
corr

# COMMAND ----------

