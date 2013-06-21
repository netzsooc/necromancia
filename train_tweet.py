#-*-encoding: utf-8 -*-
import nltk
import pickle

p=open("pos_training.txt","r")
pos_tweets = []
for i in p:
    ns = i.strip()
    x = ns,'positivo'
    pos_tweets.append(x)
#print pos_tweets

n=open("neg_training.txt","r")
neg_tweets = []
for i in n:
    ns = i.strip()
    x = ns,'negativo'
    neg_tweets.append(x)
  
#print neg_tweets

def ngramas(n, string):
##    """Toma la cadena y devuelve ngramas (caracteres si es un str palabras si
##es una lista). n es definido por el ususario"""

    ngrams = []
    i = 0
    while i + n < len(string):
        ngrams.append(string[i:i + n + 1])
        i += 1

    return ngrams

	
def trigramas(words):
    return ngramas(2, words)
	
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
	words_filtered = [e.lower() for e in trigramas(words) if len(e) >= 1]
	tweets.append((words_filtered, sentiment))
#print tweets

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)

    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()

    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))
#print word_features

f = open('data_features.pickle', 'wb')
pickle.dump(word_features, f)
f.close()

def extract_features(document):
    document_words = set(document)

    features = {}
    for word in word_features:
        features['contiene(%s)' % word] = (word in document_words)

    return features

training_set = nltk.classify.apply_features(extract_features, tweets)
#print training_set

classifier = nltk.NaiveBayesClassifier.train(training_set)

f = open('data_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

print classifier.show_most_informative_features(10)
