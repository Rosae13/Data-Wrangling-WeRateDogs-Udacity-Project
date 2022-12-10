#!/usr/bin/env python
# coding: utf-8

# # Project: Wrangling and Analyze Data

# This project is aimed to practice the skills of data wrangling using real-world data. Real-world data rarely comes clean, Data wrangling process consists of three parts: Gather, Assess and Clean. Real-world data rarely comes clean, so using Python and its libraries, we will gather data from a variety of sources and in a variety of formats, assess its quality and tidiness, then clean it. The dataset for this project is the tweet archive of Twitter user @dog_rates, also known as WeRateDogs. WeRateDogs is a Twitter account that rates people's dogs with a humorous comment about the dog.  We will document our wrangling efforts in a Jupyter Notebook, plus showcase them through analyses and visualizations using Python (and its libraries).

# ## Table of Contents:

# * Data Gathering
# 
#  <a href='#Data Gathering'>Link to the Data Gathering'</a>
#  
#  
# * Assessing
# 
#  <a href='#Assessing Data'>Link to the Assessing Data'</a>
# 
# 
# * Cleaning
# 
#  <a href='#Cleaning Data'>Link to the Cleaning Data'</a>
# 
# 
# * Storing
# 
#  <a href='#Storing Data'>Link to the Storing Data'</a>
# 
# 
# * Analysis and Visualization
# 
#  <a href='#Analyzing and Visualizing Data'>Link to the Analyzing and Visualizing Data'</a>

# <a id='Data Gathering'></a>
# ## Data Gathering
# In the cell below, gather **all** three pieces of data for this project and load them in the notebook. **Note:** the methods required to gather each data are different.
# 1. Directly download the WeRateDogs Twitter archive data (twitter_archive_enhanced.csv)

# In[1]:


#Import statements for all packages
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import tweepy
import json
from IPython.display import Image
import os
import re
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
import datetime
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# script to upgrade certificate on this Workspace
get_ipython().system('pip install --upgrade certifi ')


# In[3]:


# Load Data
twitter_archives = pd.read_csv('twitter-archive-enhanced.csv')

# Checking the firstfew values of the dataset
twitter_archives.head(2)


# 2. Use the Requests library to download the tweet image prediction (image_predictions.tsv)

# #### Image Predictions

# In[4]:


# Let's dowload the Image Predictions file programmatically:
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
image_request = requests.get(url, allow_redirects=True)

# Let's save the file
open('image_predictions.tsv', 'wb').write(image_request.content)


# In[5]:


# Read TSV file
image_predictions = pd.read_csv('image_predictions.tsv', sep = '\t')

# Checking the firstfew values of the dataset
image_predictions.head()


# 3. Use the Tweepy library to query additional data via the Twitter API (tweet_json.txt)

# #### Twitter API

# In[6]:


# Let's download file using Requests library via URL provided to Udacity Students 
url = 'https://video.udacity-data.com/topher/2018/November/5be5fb7d_tweet-json/tweet-json.txt'
response = requests.get(url)

# Save the file
with open('tweet-json.txt', mode = 'wb') as file:
    file.write(response.content)


# In[7]:


# Now, we can read downloaded txt file line by line into our Pandas DataFrame
df_list = []
with open('tweet-json.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        parsed_json = json.loads(line)
        df_list.append({'tweet_id': parsed_json['id'],
                        'retweet_count': parsed_json['retweet_count'],
                        'favorite_count': parsed_json['favorite_count']})
        
tweet_json = pd.DataFrame(df_list, columns = ['tweet_id', 'retweet_count', 'favorite_count'])

# Checking the firstfew values of the dataset
tweet_json.head()


# <a id='Assessing Data'></a>
# ## Assessing Data
# In this section, detect and document at least **eight (8) quality issues and two (2) tidiness issue**. You must use **both** visual assessment
# programmatic assessement to assess the data.
# 
# **Note:** pay attention to the following key points when you access the data.
# 
# * You only want original ratings (no retweets) that have images. Though there are 5000+ tweets in the dataset, not all are dog ratings and some are retweets.
# * Assessing and cleaning the entire dataset completely would require a lot of time, and is not necessary to practice and demonstrate your skills in data wrangling. Therefore, the requirements of this project are only to assess and clean at least 8 quality issues and at least 2 tidiness issues in this dataset.
# * The fact that the rating numerators are greater than the denominators does not need to be cleaned. This [unique rating system](http://knowyourmeme.com/memes/theyre-good-dogs-brent) is a big part of the popularity of WeRateDogs.
# * You do not need to gather the tweets beyond August 1st, 2017. You can, but note that you won't be able to gather the image predictions for these tweets since you don't have access to the algorithm used.
# 
# 

# ### Twitter

# In[8]:


twitter_archives.head()


# In[9]:


twitter_archives.source.value_counts()


# #### Quality:
# 
#     Some columns have NaN values
#     
#     Source info can not be read easily.
# 
# #### Tidiness:
# 
#     Dog type is in four columns (doggo, floofer, pupper, puppo), which suppose to be in one column.
# 
# 

# In[10]:


# Get general info
twitter_archives.info()


# In[11]:


#filter out all the rows where retweeted_status_id is null
twitter_archives[- twitter_archives.retweeted_status_id.isnull()]


# #### Quality:
# 
#  *   These aren't really valid as there are duplicated tweets and we don't want retweets.
#  
# 
#  *   in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id should be integers instead of float.
#     
#     
#  *   retweeted_status_timestamp, timestamp should be datetime instead of object which is string.
# 
# 
#  *   We may change this columns type: in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id and tweet_id to string because We don't want any operations on them.
# 
# 

# In[12]:


# Check for duplicates
twitter_archives.duplicated().sum()


# In[13]:


# Check the column labels of the DataFrame
twitter_archives.columns


# In[14]:


# Return a Series containing counts of Name rows in the DataFrame.
twitter_archives['name'].value_counts()


# #### Quality:
# 
#     There are invalid names like a, an, etc., which is less than three characters.
# 
#     There are duplicated names

# In[15]:


# Return a Series containing counts of rating_numerator rows in the DataFrame.
twitter_archives.rating_numerator.value_counts()


# #### Observation
# 
# Abnormal values in rating_numerator, e.g., 1776, 960, 666, 204, 165, etc., is inappropriate

# In[16]:


# Return a Series containing counts of rating_denominator rows in the DataFrame.

twitter_archives.rating_denominator.value_counts()


# #### Observation
# 
# Abnormal values in rating_denominator, e.g., 170, 150, 130, etc, because these ratings almost always have a denominator of 10.

# ### Image Predictions

# In[17]:


image_predictions


# In[18]:


# Get general info
image_predictions.info()


# In[19]:


# Return a random sample of items in Image Predictions Table
image_predictions.sample(5)


# In[20]:


# Testing Image
# This is an image for tweet_id: 770093767776997377
Image(url = 'https://pbs.twimg.com/media/CkjMx99UoAM2B1a.jpg')


# In[21]:


# Overall descriptive statistics
image_predictions.describe()


# In[22]:


# Check for duplicates
image_predictions.tweet_id.duplicated().sum()


# In[23]:


# Return a Series containing counts of p1 rows in the DataFrame.

image_predictions.p1.value_counts()


# In[24]:


# Return a Series containing counts of p2 rows in the DataFrame.

image_predictions.p2.value_counts()


# In[25]:


# Return a Series containing counts of p3 rows in the DataFrame.

image_predictions.p3.value_counts()


# #### Observation:
# 
#   *  Inconsistent capitalization in p1, p2 and p3 columns
#   
#   *  Many entries are not dogs, e.g., soccer_ball, cardigan, stove, pot, mailbox, shovel, banana, etc.

# ### Tweet Json

# In[26]:


tweet_json


# #### Observation:  
#     Missing data probably due to retweets in twitter_archive

# In[27]:


# Get general info
tweet_json.info()


# In[28]:


# Overall Descriptive statistics
tweet_json.describe()


# In[29]:


# Check for duplicates
tweet_json.duplicated().sum()


# ### Quality 
# 
# 1.  Some missing values in the column. Remove retweets: It is only original tweets we need.
# 
# 2.  Name column contain false names like "a".
# 
# 3.  Underscores in p1, p2 and p3 column
# 
# 4.  Inconsistent capitalization in p1, p2 and p3 columns
# 
# 5.  False predictions: predictions contain many entries that are not dogs , e.g., soccer_ball, cardigan, stove, pot,
#     mailbox, shovel, banana, etc.
# 
# 6.  The proportions in p1_conf, p2_conf and p3_conf columns should be percentages
# 
# 7. The numerator and denominator columns have invalid values and rating containing decimal numbers in numerator
# 
# 8.  Most predicted breed for each prediction level should be created.
# 
# 9.  Some column headers are not descriptive e.g jpg_url
# 
# 10.  Sources format are not readable.
# 
# 11.  timestamp is in object instead of string and datetime and date and time should be separated
# 
# 12.  Remove duplicated columns
# 
# 13.  Some columns are in inappropriate data type e.g tweet_id should be a string not an integer

# 

# 

# 

# ### Tidiness
# 
# 1.  Dog type(doggo, floofer, pupper, puppo) should be in one column
# 
# 2.  The three tables(twitter_archive, image_predictions and tweet_json) should be merged into one since they're all
#     related to the same type of observational unit according to tidy data requirements

# <a id='Cleaning Data'></a>
# ## Cleaning Data
# In this section, clean **all** of the issues you documented while assessing. 
# 
# **Note:** Make a copy of the original data before cleaning. Cleaning includes merging individual pieces of data according to the rules of [tidy data](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html). The result should be a high-quality and tidy master pandas DataFrame (or DataFrames, if appropriate).

# In[30]:


# Make copies of original pieces of data
twitter_archives_clean = twitter_archives.copy()
image_predictions_clean = image_predictions.copy()
tweet_json_clean = tweet_json.copy()


# #### Merge the clean versions of twitter_archives, image_predictions, and tweet_json dataframes 

# In[31]:


twitter_dogs_data = pd.concat([twitter_archives_clean, image_predictions_clean, tweet_json_clean], join='outer', axis=1)


# In[32]:


twitter_dogs_data.head()


# ### Issue #1:
# 
# Removing the missing values 

# #### Define:
# 
# There are some missing values in the column, so we need to remove them.

# #### Code

# In[33]:


# checking for missing values in the dataframe
twitter_dogs_data.isna().sum()


# In[34]:


# This gives you a percentage value of missing values
(twitter_dogs_data.isna().sum()/twitter_dogs_data.shape[0])*100

# when there is too much percentage of missing values you drop the columns


# In[35]:


# Let's drop missing values
twitter_dogs_data.dropna(axis=1).isna().sum()


# #### Test

# In[36]:


twitter_dogs_data.info()


# #### Removing retweets

# #### Define:
# 
# We only need original tweets

# #### Code

# In[37]:


twitter_dogs_data = twitter_dogs_data[twitter_dogs_data.retweeted_status_id.isnull()]
twitter_dogs_data = twitter_dogs_data[twitter_dogs_data.retweeted_status_user_id.isnull()]
twitter_dogs_data = twitter_dogs_data[twitter_dogs_data.retweeted_status_timestamp.isnull()]


# #### Test

# In[38]:


twitter_dogs_data.info()


# #### Let's drop columns not needed

# #### Code

# In[39]:


# We will remove retweet and a column we won't use
twitter_dogs_data = twitter_dogs_data.drop(['in_reply_to_status_id', 'in_reply_to_user_id',
                                                      'retweeted_status_id', 'retweeted_status_user_id',
                                                      'retweeted_status_timestamp', 'expanded_urls'], axis = 1)


# #### Test

# In[40]:


twitter_dogs_data.info()


# ### Issue #2:
# 
# Here, we will fix the name column. It contains false names like "a"

# #### Define:
# 
# Change lowercase names to None as they are wrong.

# #### Code

# In[41]:


#filtering out rows where there is the word 'named' in the text and the name is in lowercase - these are probably the rows where 
#the names are incorrect and then creating a list out of their corresponding indices
index_list = twitter_dogs_data.loc[twitter_dogs_data.text.str.contains('named') & 
                                              twitter_dogs_data.name.str.islower()].index.tolist()
all_indices = twitter_dogs_data.index.tolist()
for e in all_indices:
    if e in index_list:
        for ele in list(range(len(index_list))):
            #creating a list out of all such text values that contain the word 'named' and the corresponding name value 
            #is in lowercase            
            text_list = (twitter_dogs_data.loc[twitter_dogs_data.text.str.contains('named') & 
                                                 twitter_dogs_data.name.str.islower()].text).tolist()
            #finding the index position in every text value in the list where 'named' occurs
            num = text_list[ele].find('named')
            #using this index position to extract a particular pattern of dog name out of the text and then assign it to the
            #corresponding name value
            x = twitter_dogs_data.loc[twitter_dogs_data.index == e, 'text'].str[num+6:].str.extract(r'([A-Z][a-z]+)',
                                                                                                          expand = True)[0]
            twitter_dogs_data.loc[twitter_dogs_data.index == e, 'name'] = x
            break


# In[42]:


#replacing the remaining 'None's in the name column by NaN values
twitter_dogs_data.name = twitter_dogs_data.name.replace('None', np.nan)
twitter_dogs_data.name.head()


# #### Test

# In[43]:


twitter_dogs_data.loc[twitter_dogs_data.text.str.contains('named')]


# In[44]:


twitter_dogs_data[twitter_dogs_data.name == 'None'] # This is for checking for none values
twitter_dogs_data[twitter_dogs_data.name == 'Nan'] # Also checked for missing values and found none


# ### Issue #3:
# 
# Replacing the Underscores

# #### Define:
# 
# Replace the underscores in the p1, p2 and p3 columns by spaces

# #### Code

# In[45]:


# Replacing using the replace function
twitter_dogs_data.p1 = twitter_dogs_data.p1.str.replace('_',' ')
twitter_dogs_data.p2 = twitter_dogs_data.p2.str.replace('_',' ')
twitter_dogs_data.p3 = twitter_dogs_data.p3.str.replace('_',' ')


# #### Test

# In[46]:


twitter_dogs_data[['p1', 'p2', 'p3']].head()


# ### Issue #4:
# 
# Inconsistent capitalization in p1, p2 and p3 columns

# #### Define:
# 
# Fixing inconsistent capitalization in p1, p2 and p3 columns

# #### Code

# In[47]:


# Fix capitalization by using the str.title function
twitter_dogs_data.p1 = twitter_dogs_data.p1.str.title()
twitter_dogs_data.p2 = twitter_dogs_data.p2.str.title()
twitter_dogs_data.p3 = twitter_dogs_data.p3.str.title()


# #### Test

# In[48]:


twitter_dogs_data[['p1', 'p2', 'p3']]


# ### Let's remove the NaNs in p1, p2, p3 columns

# #### Code

# In[49]:


dog_breed = twitter_dogs_data[['p1', 'p2', 'p3']]
dog_breed.head()


# In[50]:


dog_breed.shape


# In[51]:


dog_breed.isna().sum()


# In[52]:


dog_breed.dropna(inplace=True)


# #### Test

# In[53]:


dog_breed.isna().any()


# In[54]:


dog_breed.sample(10)


# In[55]:


dog_breed.info()


# In[56]:


twitter_dogs_data.info()


# ### Issue #5:
# 
# False predictions: predictions contain many entries that are not dogs

# #### Define:
# 
# Drop each of the false row prediction

# #### Code

# In[57]:


false_predictions = ~((twitter_dogs_data.p1_dog) | (twitter_dogs_data.p2_dog) | (twitter_dogs_data.p3_dog))
false_predictions_dog = twitter_dogs_data[false_predictions].index.tolist()


# In[58]:


len(twitter_dogs_data[false_predictions])


# In[59]:


twitter_dogs_data.drop(false_predictions_dog, inplace = True)


# In[60]:


twitter_dogs_data = twitter_dogs_data.reset_index(drop = True)


# #### Test

# In[61]:


false_predictions = ~((twitter_dogs_data.p1_dog) | (twitter_dogs_data.p2_dog) | (twitter_dogs_data.p3_dog))

len(twitter_dogs_data[false_predictions])


# ### Issue #6:
# 
# The proportions in p1_conf, p2_conf and p3_conf columns should be percentages

# #### Define:
# 
# Convert the proportions in the p1_conf, p2_conf and p3_conf columns into percentages

# In[62]:


#using apply, multiplying 100 to each column value in each row
twitter_dogs_data.p1_conf = twitter_dogs_data.p1_conf.apply(lambda x: round(x*100, 2))
twitter_dogs_data.p2_conf = twitter_dogs_data.p2_conf.apply(lambda x: round(x*100, 2))
twitter_dogs_data.p3_conf = twitter_dogs_data.p3_conf.apply(lambda x: round(x*100, 2))


# In[63]:


twitter_dogs_data.head()


# ### Issue #7:
# 
# The numerator and denominator columns have invalid values.

# #### Define:
# 
# Fix rating numerator and denominators that are not ratings

# #### Code

# In[64]:


tmp_rating = twitter_dogs_data[twitter_dogs_data.text.str.contains( r"(\d+\.?\d*\/\d+\.?\d*\D+\d+\.?\d*\/\d+\.?\d*)")].text

for i in tmp_rating:
    x = twitter_dogs_data.text == i
    column_1 = 'rating_numerator'
    column_2 = 'rating_denominator'
    twitter_dogs_data.loc[x, column_1] = re.findall(r"\d+\.?\d*\/\d+\.?\d*\D+(\d+\.?\d*)\/\d+\.?\d*", i)
    twitter_dogs_data.loc[x, column_2] = 10


# #### Test

# In[65]:


twitter_dogs_data[twitter_dogs_data.text.isin(tmp_rating)]


# ### Rating containing decimal numbers in numerator.

# #### Define:
# 
# Clean decimal values in rating numerators.

# #### Code

# In[66]:


twitter_dogs_data[twitter_dogs_data.text.str.contains(r"(\d+\.\d*\/\d+)")]


# In[67]:


ratings = twitter_dogs_data.text.str.extract('((?:\d+\.)?\d+)\/(\d+)', expand=True)
ratings


# In[68]:


#Convert the null values to None type
twitter_dogs_data['rating_numerator'] = ratings[0]


# #### Test

# In[69]:


twitter_dogs_data[twitter_dogs_data.text.str.contains(r"(\d+\.\d*\/\d+)")]


# ### Issue #8:
# 
# Most predicted breed for each prediction level should be created.

# #### Define:
# 
# Create top accurate predicted dog breed in a column and drop p1,p2,p3 related columns

# #### Code

# In[70]:


columns = ['p1_dog','p2_dog','p3_dog']
pred = []
preds = []

for index in range(twitter_dogs_data.shape[0]):
    for col in columns:
        if twitter_dogs_data.loc[index,col] == True:
            pred.append(col)
    preds.append(pred)
    pred = []


# In[71]:


corresponding_columns = {'p1_dog': ['p1','p1_conf'], 'p2_dog': ['p2','p2_conf'], 'p3_dog': ['p3','p3_conf']}
prediction1 = []
conf1 = []

for index in range(twitter_dogs_data.shape[0]):
    prediction1.append(twitter_dogs_data.loc[index,corresponding_columns[preds[index][0]][0]])
    conf1.append(twitter_dogs_data.loc[index,corresponding_columns[preds[index][0]][1]])

prediction1 = pd.Series(prediction1)
conf1 = pd.Series(conf1)


# In[72]:


twitter_dogs_data = pd.concat([twitter_dogs_data, prediction1,conf1], axis=1, join = 'outer')


# In[73]:


twitter_dogs_data.rename(columns={0: "dog_breed_prediction", 1: "confidence_percentage"}, inplace = True)


# In[74]:


twitter_dogs_data.drop(columns=['p1', 'p2','p3','p1_conf','p1_conf','p2_conf','p3_conf','p1_dog','p2_dog','p3_dog'], inplace = True)


# #### Test

# In[75]:


twitter_dogs_data.info()


# In[76]:


twitter_dogs_data.head()


# ### Issue #9:
# 
# Some column headers are not descriptive

# #### Define:
# 
# Change column headers to be more readable column names e.g, the column name of jpg_url for better viewing.

# #### Code

# In[77]:


twitter_dogs_data.rename(columns={'jpg_url': 'image_url', 'name': "dog_name"}, inplace = True)


# #### Test

# In[78]:


twitter_dogs_data.head()


# ### Issue #10:
# 
# Sources format are not readable.

# #### Define:
# 
# Make 'source' column clean and readable.

# #### Code

# In[79]:


twitter_dogs_data['source'] = twitter_dogs_data['source'].apply(lambda x: re.findall(r'>(.*)<', x)[0])


# #### Test

# In[80]:


twitter_dogs_data.head()


# ### Issue #11:
# 
# timestamp is in object instead of string and datetime

# #### Define:
# 
# timestamp should be in datetime format and date and time should be separated

# #### Code

# In[81]:


twitter_dogs_data.timestamp = pd.to_datetime(twitter_dogs_data.timestamp, yearfirst = True)


# In[82]:


#using the apply function, applying the strftime function to each value of the timestamp column in each row
twitter_dogs_data['date'] = twitter_dogs_data['timestamp'].apply(lambda x: x.strftime('%d-%m-%Y'))
twitter_dogs_data['time'] = twitter_dogs_data['timestamp'].apply(lambda x: x.strftime('%H:%M:%S'))

#changing datatype of the date column to datetime
twitter_dogs_data.date = pd.to_datetime(twitter_dogs_data.date, dayfirst = True)


# In[83]:


#Now, let's drop the timestamp column
twitter_dogs_data = twitter_dogs_data.drop('timestamp', axis = 1)


# #### Test

# In[84]:


twitter_dogs_data.info() 


# ### Issue #12:
# 
# Duplicated Columns

# #### Define:
# 
# Remove duplicated columns

# #### Code

# In[85]:


twitter_dogs_data = twitter_dogs_data.loc[:,~twitter_dogs_data.columns.duplicated()]


# #### Test

# In[86]:


twitter_dogs_data.info()


# ### Dog Type

# #### Define:
# 
# Dog type(doggo, floofer, pupper, puppo) should be in one column

# #### Code

# In[87]:


# Checking for multiple stages
twitter_dogs_data[['doggo','floofer','pupper','puppo']].sum(axis=1).unique()


# In[88]:


twitter_dogs_data['multiple_stages'] = twitter_dogs_data[['doggo','floofer','pupper','puppo']].sum(axis=1) == 2
multiple_stages = twitter_dogs_data.query('multiple_stages == True')[['text', 'doggo','floofer','pupper','puppo']]
multiple_stages.shape


# In[89]:


twitter_dogs_data['dog_type'] = twitter_dogs_data[
    ['doggo', 'floofer', 'pupper', 'puppo']].apply(lambda x: ', '.join(x), axis=1)


# In[90]:


twitter_dogs_data.drop(columns = ['doggo', 'floofer','pupper','puppo'], inplace = True)


# In[91]:


twitter_dogs_data = twitter_dogs_data.replace(regex=r'(None,? ?)', value='').replace(regex=r'(, $)', value='')


# In[92]:


twitter_dogs_data = twitter_dogs_data.replace(regex=r'', value= np.nan)


# #### Test

# In[93]:


twitter_dogs_data.dog_type.value_counts()


# In[94]:


twitter_dogs_data.info()


# In[95]:


twitter_dogs_data.head()


# ### Issue #13:
# 
# Data Types

# #### Define:
# 
# Change datatypes 

# #### Code

# In[96]:


twitter_dogs_data['tweet_id'] = twitter_dogs_data['tweet_id'].astype(str)

twitter_dogs_data['dog_type'] = twitter_dogs_data['dog_type'].astype('category')

twitter_dogs_data['source'] = twitter_dogs_data['source'].astype('category')

twitter_dogs_data['rating_numerator'] = twitter_dogs_data['rating_numerator'].astype(float)

twitter_dogs_data['rating_denominator'] = twitter_dogs_data['rating_denominator'].astype(float)


# #### Test

# In[97]:


twitter_dogs_data.dtypes


# <a id='Storing Data'></a>
# ## Storing Data
# Save gathered, assessed, and cleaned master dataset to a CSV file named "twitter_archive_master.csv".

# In[98]:


twitter_dogs_data.to_csv('twitter_archive_master.csv', encoding = 'utf-8', index=False)


# <a id='Analyzing and Visualizing Data'></a>
# ## Analyzing and Visualizing Data
# In this section, analyze and visualize your wrangled data. You must produce at least **three (3) insights and one (1) visualization.**

# In[99]:


df = pd.read_csv('twitter_archive_master.csv')


# In[100]:


# Get general info
df.info()


# In[101]:


# Convert columns to their appropriate types and set the timestamp as an index
df['tweet_id'] = df['tweet_id'].astype(object)
df['source'] = df['source'].astype('category')
df['multiple_stages'] = df['multiple_stages'].astype('category')
df['dog_type'] = df['dog_type'].astype('category')


# In[102]:


df.info()


# In[103]:


df['multiple_stages'].value_counts()


# In[104]:


df.drop(columns = ['multiple_stages'], inplace = True)


# In[105]:


df.info()


# <a id='Insights and Visualization'></a>
# ### Insights and Visualization:
# 
# 1.  The most popular dog type:
# 
#   -   By rating ratio
#   -   By favorite count
#   -   By retweet count
# 
# 
# 2.  The most popular dog breed:
# 
#   -   By rating ratio
#   -   By favorite count
#   -   By retweet count
# 
# 
# 3.  The relationship between retweets and favorites

# ### The most popular dog type

# In[106]:


df['rating_ratio'] = df['rating_numerator'] / df['rating_denominator']


# In[107]:


df['dog_type'].value_counts()


# In[108]:


# Remove missing stages, the stages that only have one sample size, and outliers
stage = df.query('dog_type !="" & dog_type != "doggo, puppo" & dog_type != "doggo, floofer"')
stage['dog_type'].value_counts()


# ### By Rating Ratio

# In[109]:


stage.groupby('dog_type')['rating_ratio'].describe()


# ### By Favorite Count

# In[110]:


stage.groupby('dog_type')['favorite_count'].describe()


# In[111]:


#Now, let's plot a boxplot
plt.figure(figsize=(10,8))
sns.set(style="darkgrid")
sns.boxplot(x='dog_type', y="favorite_count", data=stage).set_title('Dog Type by Favorite Count')
plt.savefig('stage_fav.png');


# ### By Retweet Counts

# In[112]:


stage.groupby('dog_type')['retweet_count'].describe()


# In[113]:


#Now, let's plot a boxplot
plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.boxplot(x='dog_type', y="retweet_count", data=stage).set_title('Dog Type by Retweet Count')
plt.savefig('stag_fav.png');


# #### As you can see, the most popular dog type is Pupper

# ### The most popular dog breed

# In[114]:


df.dog_breed_prediction.value_counts().head(5)


# In[115]:


# Find out the top 5 breed that has the most images
array = ['Golden_Retriever','Labrador_Retriever','Chihuahua','Pembroke','Pug']
new_df = df.loc[df['dog_breed_prediction'].isin(array)]


# ### By Rating Ratio

# In[116]:


new_df.groupby('dog_breed_prediction').rating_ratio.describe()


# In[117]:


new_df.groupby('dog_breed_prediction')['dog_type'].value_counts()


# ### By Favorite Count

# In[118]:


new_df.groupby('dog_breed_prediction')['favorite_count'].describe()


# In[119]:


plt.figure(figsize=(10,8))
sns.boxplot(x='dog_breed_prediction', y="favorite_count", data=new_df).set_title('Dog Breed by Favorite Count')
plt.savefig('breed_fav.png');


# ### By Retweet Counts

# In[120]:


new_df.groupby('dog_breed_prediction')['retweet_count'].describe()


# In[121]:


plt.figure(figsize=(10,8))
sns.boxplot(x='dog_breed_prediction', y="retweet_count", data=new_df).set_title('Dog Breed by Retweet Count')
plt.savefig('breed_retw.png');


# #### Now, you see that the most popular dog breed is Chihuahua

# ### The relationship between retweets and favorites

# In[122]:


#Retweets vs. Favorites

df.plot(kind='scatter',x='favorite_count',y='retweet_count', alpha = 0.5)
plt.xlabel('Favorites')
plt.ylabel('Retweets')
plt.title('Retweets and Favorites Scatter Plot')

plt.savefig('Retweets_vs_Favorites.png', bbox_inches='tight')


# #### Here, we can see that Retweets are positively correlated with Favorites.

# ## References:

# *   https://stackoverflow.com/questions/44327999/python-pandas-merge-multiple-dataframes/44338256
#     
# *   https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
#     
# *   Data Analysis Nanodegree/Data Wrangling/Lesson 3: Assessing Data/Concepts 4-18
#     
# *   https://video.udacity-data.com/topher/2018/November/5be5fb7d_tweet-json/tweet-json.txt
#     
# *   https://stackoverflow.com/questions/44594945/pandas-str-replace-regex-application
#     
# *   https://pandas.pydata.org
#     
# *   https://www.w3resource.com/pandas/series/series-str-islower.php
#     
# *   https://docs.python.org
#     
# *   https://thepythonguru.com
#     
# *   https://ipython.org
# 
# *   https://sebastianraschka.com/Articles/2014_ipython_internal_links.html#bottom    
# 
# *  https://medium.com/@sambozek/ipython-er-jupyter-table-of-contents-69bb72cf39d3
