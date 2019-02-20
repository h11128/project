import tweepy
import re
from tweepy import OAuthHandler
from textblob import TextBlob
import json
import sys
import jsonpickle
import os

class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):

        try:
            self.auth = tweepy.OAuthHandler("oDGPSLRthhVvbrk22aF5vgldV", "IigqG87FbIuhcZpHUxuq4O2adgrFuX3eg8dwqRNulffNrzYifP")
            self.auth.set_access_token("355356323-rPDfRC56bekq7aHlYh0iXYcxxjqPiiJuD2ngr2Fa", "5LiGqpUtkiZKrbOpOdJvekxfgssDaqqtVHfa2c3qpjrAZ")
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

def main():
    api = TwitterClient()
    searchQuery = 'Hurricane'  # this is what we're searching for
    maxTweets = 10000000 # Some arbitrary large number
    tweetsPerQry = 100  # this is the max the API permits
    fName = 'tweets.txt' # We'll store the tweets in a text file.
    sinceId = None
    max_id = -1
    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        tweets = []
        i = 0
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,max_id=str(max_id - 1),since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    if tweet.user.location:
                        i = i+10
                        print(i)
                        parsed_tweet = {}
                        parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                        parsed_tweet['location'] = tweet.user.location
                        parsed_tweet['created_at'] = tweet.created_at
                        if tweet.retweet_count > 0:
                            if parsed_tweet not in tweets:
                                tweets.append(parsed_tweet)
                        else:
                            tweets.append(parsed_tweet)
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
        for tweet in tweets:
            f.write(str(tweet)+"\n")
            except tweepy.TweepError as e:
                print("some error : " + str(e))
                break
    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

if __name__ == "__main__":
    main()
