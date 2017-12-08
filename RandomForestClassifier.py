import pandas as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler


def features_from_tweet(data1, x_train, x_test):
	#features = data1[["retweet_count", "tweet_count", "fav_number"]]
	#print("features",features)
	train_features = x_train[["retweet_count", "tweet_count", "fav_number"]]
	test_features = x_test[["retweet_count", "tweet_count", "fav_number"]]
	#print("train_features: ",train_features)
	scaler = StandardScaler().fit(train_features)
	print("scaler :",scaler)
	return (scaler.transform(train_features), scaler.transform(test_features))


def label_encoding(train_data,test_data):
	label_encoder = LabelEncoder()
	y_train = label_encoder.fit_transform(train_data.fillna('unknown'))
	y_test = label_encoder.fit_transform(test_data.fillna('unknown'))
	print("Train:",y_train)
	print("Test:",y_test)
	print("Classes:",label_encoder.classes_)
	return (y_train,y_test,label_encoder.classes_)

def normalizing_text(data):
    # Removing non-ASCII characters, double spaces, special characters, urls from passed data
    data_new = re.sub('[^\x00-\x7F]+', ' ',data)
    data_new = re.sub('\s+', ' ', data_new)
    data_new = re.sub('[?!+%{}:;.,"\'()\[\]_]', '', data_new)
    data_new = re.sub('https?:\/\/.*[\r\n]*', ' ',data_new)
    return data_new

def tfidf_extracter(dataset, X_train,X_test):
	tfidf_feature = TfidfVectorizer(strip_accents="unicode")
	dataset["normalized_text"] = [normalizing_text(data) for data in dataset["text"]]
	dataset["normalized_desc"] = [normalizing_text(data) for data in dataset["description"].fillna("")]


	#train_text = dataset.ix[X_train, :]["normalized_text"]
	#train_desc = dataset.ix[X_train, :]["normalized_desc"]

	X_train["normalized_text"] = [normalizing_text(data) for data in X_train["text"]]
	X_train["normalized_desc"] = [normalizing_text(data) for data in X_train["description"].fillna("")]

	train_text = X_train["normalized_text"]
	train_description = X_train["normalized_desc"]

	tfidf = tfidf_feature.fit(train_text.str.cat(train_description,sep=' '))

	print("done first")

	X_train_transformed = tfidf.transform(train_text.str.cat(train_description,sep=' '))
	#X_train_new = X_train
	#X_train_transformed = p.DataFrame(X_train_transformed.toarray(), columns=tfidf.get_feature_names())
	#X_train_new.append(X_train_transformed)

	X_test["normalized_text"] = [normalizing_text(data) for data in X_test["text"]]
	X_test["normalized_desc"] = [normalizing_text(data) for data in X_test["description"].fillna("")]
	test_text = X_test["normalized_text"]
	test_description = X_test["normalized_desc"]
	X_test_transformed = tfidf.transform(test_text.str.cat(test_description,sep=' '))
	#X_test_new = X_test
	#X_test_transformed = p.DataFrame(X_test_transformed.toarray(), columns=tfidf.get_feature_names())
	#X_test_new.append(X_test_transformed)

	print("X",dataset.shape)
	#print("X training: ",X_train_new.shape)
	#print("X test",X_test_new.shape)

	#df1 = p.DataFrame(X_train_transformed.toarray(), columns=tfidf.get_feature_names())

	#print("data",df1.dtypes)

	#return X_train_new,X_test_new
	return X_train_transformed,X_test_transformed



def dataPreprocessing():
	# fill missing values
	# label encoding of target variables
	# Check variable improtance
	# normalizing and vectorizing the text and description
	# Scaling
	return 0




if __name__=="__main__":
    #import dataset
    twitter_user_dataset=p.read_csv("C:/Users/Jobin/Documents/GitHub/mltwitter/gender-classifier-DFE-791531.csv",encoding='latin1')
    #print(twitter_user_dataset.dtypes)
    #preprocessing of data


    #assigning target to y variable and indeendent variables to x
    x = twitter_user_dataset
    y = twitter_user_dataset['gender']
    x = x.drop(['gender','_unit_id','_golden','_unit_state','_trusted_judgments','_last_judgment_at','profile_yn','profile_yn:confidence','created','name','profile_yn_gold','profileimage','tweet_created','tweet_id','tweet_location','user_timezone','tweet_coord'],axis=1)

    #splitting of data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    X_train_new, X_test_new = tfidf_extracter(twitter_user_dataset, x_train, x_test)
    feature_tweet_train, feature_tweet_test = features_from_tweet(twitter_user_dataset, x_train, x_test)

    X_train_n = hstack((X_train_new,feature_tweet_train))
    X_test_n = hstack((X_test_new,feature_tweet_test))

    print("New train",X_train_n.col)


    #encoding
    y_train,y_test,label_encoder_classes = label_encoding(y_train,y_test)

    #model creation
    classifier = RandomForestClassifier(n_estimators=1000,n_jobs=2,random_state=0)
    classifier.fit(X_train_n,y_train)

    y_pred = classifier.predict(X_test_n)

    #calculating the accuracy
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
