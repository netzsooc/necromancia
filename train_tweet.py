#-*-encoding: utf-8 -*-
from nltk import FreqDist
from tf import ntf
import pickle


def set_labeled_training_text(file_name, label):
    labeled_set = []

    with open(file_name) as my_file:
        for line in my_file:
            line = line.strip()
            label_set.append((line, category))

    return label_set


def ngramas(n, string):
    """Toma la cadena y devuelve ngramas (caracteres si es un str palabras si
es una lista). n es definido por el ususario"""

    ngrams = []
    i = 0
    while i + n < len(string):
        ngrams.append(string[i:i + n])
        i += 1

    return ngrams


#def trigramas(words):
#    return ngramas(3, words)


#def bigramas(words):
#    return ngramas(2, words)


def get_words_in_tweets(labeled_tweets):
    all_words = []
    for (words,_) in tweets:
      all_words.extend(words)

    return all_words


def get_word_features(wordlist):
    wordlist = FreqDist(wordlist)
    word_features = wordlist.keys()

    return word_features


pos_tweets = set_labeled_training_text("pos_training.txt", "positivo")
neg_tweets = set_labeled_training_text("neg_training.txt", "positivo")

tweets = []

for (tweet, sentiment) in pos_tweets + neg_tweets:
    tweets.append((ngramas(3, tweet), sentiment))

word_features = get_word_features(get_words_in_tweets(tweets))

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
