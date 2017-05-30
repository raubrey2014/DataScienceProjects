#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file      twitter_sentiment_analysis.py
author    Ryan Aubrey <rma7qb@virginia.edu>
version   1.0
date      May 29, 2017


brief     Using Twitter API to analyze sentiment of tweets concerning specific topics
usage     python gender_classifier.py
"""
import sys
import tweepy
from textblob import TextBlob

#Authenticate with Twitter Developer services with custom app keys
consumer_key = 'trlG82ZLjOuYH7gAf5UrfgXn6'
consumer_secret = 'kArXxh3fwb8MQOzcweN5YmIxWlw5eepyh0SMdCjVqNlPMpzBJa'

access_token = '3122866047-RZ03mqG835mtm4jc41VxEIn5mtbLgYkQpcO95ao'
access_token_secret = 'ZbeiSd7iZTwCzpnmIJFDOAb13d1hmMU1fZYikZGA9lCmJ'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#What topic are we analyzing?
search_string = ""

if len(sys.argv) > 0:
	#Set the search topic to the first command line argument if there
	search_string = sys.argv[1]

else:
	#default to checking sentiment of Pres. Zuck.
	search_string = "Mark Zuckerberg president" 

#Generate our collection of SearchResult Objects, format printed below
public_tweets = api.search(search_string, 'en', 'ja', 100)

#['_api', '_json', 'created_at', 'id', 'id_str', 'text', 
#'truncated', 'entities', 'metadata', 'source', 'source_url', 
#'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 
#'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'author', 'user', 'geo', 
#'coordinates', 'place', 'contributors', 'retweeted_status', 'is_quote_status', 
#'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'lang']

#Used for average calculations
average_polarity = 0
average_subjectivity = 0

for tweet in public_tweets:
	
	if tweet.metadata['result_type'] != 'recent':
		#Only concerned with recent tweets
		pass
	else:
		#Recent tweet found! Print out some of its data
		print("\n***\n")
		print("Created At:", tweet.created_at)
		print("ID:", tweet.id)
		print("ID String:", tweet.id_str)
		print("Text:", tweet.text)
		print("Source:", tweet.source)
		print("Source URL:", tweet.source_url)

		#Use TextBlob for sentiment analysis, produces polarity, subjectivity
		analysis = TextBlob(tweet.text)
		print("\nSENTIMENT:", analysis.sentiment) 

		#Continue average calculations
		average_polarity += analysis.sentiment.polarity
		average_subjectivity += analysis.sentiment.subjectivity

if len(public_tweets) > 0:
	average_subjectivity /= len(public_tweets)
	average_polarity /= len(public_tweets)
	print("**************************************************************")
	print("**************************************************************")
	print("\n")
	#Polarity is -1 (very negative) to 1 (very positive)
	print("Average Polarity:", average_polarity) 
	#Subjectivity is 0 (very objective)to 1 (very subjective)
	print("Average Subjectivity:", average_subjectivity) 
	print("\n")
	print("**************************************************************")
	print("**************************************************************")
