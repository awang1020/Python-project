import nltk
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import pandas as pd
import numpy as np

french_stopwords = nltk.corpus.stopwords.words('french')
mots = set(line.strip() for line in open('dictionnaire.txt'))
lemmatizer = FrenchLefffLemmatizer()


def French_Preprocess_listofSentence(listofSentence):
    preprocess_list = []
    for sentence in listofSentence:
        sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])

        sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())

        tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)

        words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]

        words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

        sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in mots or not w.isalpha())

        preprocess_list.append(sentence_clean)

    return preprocess_list

lst = ['C\'est un test pour lemmatizer',
       'plusieurs phrases pour un nettoyage',
       'eh voilà la troisième !']
french_text = pd.DataFrame(lst, columns =['text'])

french_preprocess_list = French_Preprocess_listofSentence(french_text['text'])

print('Phrase de base : '+lst[1])
print('Phrase nettoyée : '+french_preprocess_list[1])

