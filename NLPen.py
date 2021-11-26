#  Preprocessing NLP (Tuto pour nettoyer rapidement le texte)

import numpy as np
import pandas as pd
import nltk
import string
from nltk.stem import WordNetLemmatizer

train_data = pd.read_csv('train.csv')

print(train_data.head())


#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('words')
#nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()


def Preprocess_listofSentence(listofSentence):
    preprocess_list = []
    for sentence in listofSentence:
        sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])

        sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())

        tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)

        words_w_stopwords = [i for i in tokenize_sentence if i not in stopwords]

        words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

        sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in words or not w.isalpha())

        preprocess_list.append(sentence_clean)

    return preprocess_list

preprocess_list = Preprocess_listofSentence(train_data['text'])
print('Phrase de base : '+train_data['text'][2])
print('Phrase nettoyée : '+preprocess_list[2])