import tweepy
import re
from tweepy import OAuthHandler
from textblob import TextBlob

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

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity <= 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, counts):
        tweets = []
        print("asdasd")
        try:

            max_tweets = counts
            fetched_tweets = [status for status in tweepy.Cursor(self.api.search, q=query).items(max_tweets)]
            print("get tweets")
            print(fetched_tweets)
            """
            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return tweets
            """
        except tweepy.TweepError as e:
            print(str(e))

def main():
    api = TwitterClient()
    tweets = api.get_tweets(query = 'Hurricane', counts = 10)
    if tweepy.TweepError:
        return tweepy.TweepError
    """Get percentage of tweets"""
    """ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    print("Neutral tweets percentage: {} % \ ".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))

	with open("ntweets.txt", "w") as file:
		file.write(output)

    i = 0
    for tweet in ntweets:
        i = i+1
        if i % 10 == 0:
            print(i,": ")
            print(tweet.text)"""

if __name__ == "__main__":
    main()
