from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
from nltk.stem import WordNetLemmatizer

example_sent = """This is a sample SENTENCE3 5 67 ,showing off the stop words filtration."""

stop_words = set(stopwords.words('english')) #tout les mots qui ne sont pas utiles à la phrase (determinants..)

word_tokens = word_tokenize(example_sent)
#print(stop_words)
print(word_tokens)

stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()

#filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

sentence_w_punct = "".join([i.lower() for i in example_sent if i not in string.punctuation]) #enlever la ponctuation
sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit()) #enlever les chiffres
tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num) # split les mots sans espaces

#words_w_stopwords = []
#for i in tokenize_sentence :
#    if i not in stopwords :
#        words_w_stopwords.append(i)


words_w_stopwords = [i for i in tokenize_sentence if i not in stopwords]
words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)
sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in words or not w.isalpha())

print(sentence_w_punct)
print(sentence_w_num)
print(tokenize_sentence)
print(words_w_stopwords)
print(words_lemmatize)
print(sentence_clean)
#print(word_tokens)
#print(filtered_sentence)