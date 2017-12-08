import pandas as p

#import dataset
twitter_user_dataset=p.read_csv("C:/Users/Jobin/Documents/GitHub/mltwitter/gender-classifier-DFE-791531.csv",encoding='latin1')
dflen = len(twitter_user_dataset)
print("before drop ", dflen)
df_unknown = twitter_user_dataset[twitter_user_dataset.gender == 'unknown']
twitter_user_dataset = twitter_user_dataset[twitter_user_dataset.gender != 'unknown']
dflen = len(df_unknown)
print("after drop ", dflen)
dflen = len(twitter_user_dataset)
print("after drop ", dflen)
