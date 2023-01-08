import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Lire le fichier texte
text = open('read.txt',encoding="utf-8").read()

# Convertir le texte en lower case
lower_case = text.lower()

# Supprimer les ponctuations

# str1 : spécifie les caracteres a remplacer
# str2 : spécifie les caractères qui remplacent str1
# str3 : specifie les caracteres a supprimer
cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))
# print(cleaned_text)

# Tokenizaiton : split le text en mot
tokenized_words = word_tokenize(cleaned_text,"english")

# stop_words supprime les mots qui n'ajoute pas de sens à l'analyse


# notre phrase contenant les mots a analyser
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)
# print(final_words)

# les emotions presents dans le text final
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clean_line = line.replace('\n','').replace(',','').replace("\'",'').strip()
        word, emotion = clean_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)

# print(emotion_list)
w = Counter(emotion_list)
# print(w)

# Fonction qui retourne le score du sentiment de la phrase
def sentiment_analyze(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        print("Negative sentiment")
    elif pos > neg:
        print("Positive sentiment")
    else:
        print("Neutral sentiment")

print(sentiment_analyze(cleaned_text))

fig, ax1 = plt.subplots()
ax1.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
