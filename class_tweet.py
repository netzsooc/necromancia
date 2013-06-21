# -*- encoding: utf-8 -*-
import nltk
import pickle
import sys

f= open('data_classifier.pickle')
classifier = pickle.load(f)
f.close()
#print classifier.show_most_informative_features(32)

f= open('data_features.pickle')
word_features = pickle.load(f)
f.close()

my_file = open(sys.argv[1], 'r')
eval_tweets = []
for i in my_file:
    ns = i.strip()
    x = ns
    eval_tweets.append(x)
  
n_tweets = 0
for tweet in eval_tweets:
	n_tweets = n_tweets + 1

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
	
def extract_features(document):
    document_words = set(document)

    features = {}
    for word in word_features:
        features['contiene(%s)' % word] = (word in document_words)

    return features

#pos_n = 0
#neg_n = 0
#for tweet in eval_tweets:
	#print tweet
#	tweet_proc = [e.lower() for e in trigramas(tweet) if len(e) >= 1]
#	valor = classifier.classify(extract_features(tweet_proc))

#	print valor,':  ', tweet
		
#	if valor == 'positivo':
#		pos_n = pos_n + 1
#	if valor == 'negativo':
#		neg_n = neg_n +1
		
#print 'Accuracy'
#negativ = neg_n / n_tweets
#positiv = pos_n / n_tweets
#print 'Ac_neg: ', negativ, 'Ac_pos: ', positiv

#Para hacerlo desde raw_input
tweet = raw_input('entra tweet: ')
#print tweet
tweet = [e.lower() for e in trigramas(tweet) if len(e) >= 1]
#print tweet
print classifier.classify(extract_features(tweet))
