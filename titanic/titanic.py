import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

titanic = pd.read_csv("titanic.csv")
#print(titanic.info())
# On sélectionne les données recherchées
titanic.drop('PassengerId',axis=1, inplace=True)
titanic.drop('Name',axis=1, inplace=True)
titanic.drop('SibSp',axis=1, inplace=True)
titanic.drop('Parch',axis=1, inplace=True)
titanic.drop('Ticket',axis=1, inplace=True)
titanic.drop('Cabin',axis=1, inplace=True)
titanic.drop('Embarked',axis=1, inplace=True)
titanic.dropna()
#print(titanic.info())
# print(titanic["Sex"].value_counts())
# print(titanic["Survived"].value_counts())

#On visualise le nombre d'hommes et de femmes, le nombre de survivant sur titanic 

titanic.plot(kind="scatter", x="Sex", y="Age")
plt.show()

#Correlation entre classe sociale et survivre : 
sns.FacetGrid(titanic, hue="Survived", size=5) \
    .map(sns.kdeplot, "Pclass") \
    .add_legend()
plt.show()

#La 1ere classe avait plus de chance de survie 
#La 3e classe avait moins de chance de survie

sns.violinplot(x="Pclass", y="Survived", data=titanic, size=6)
plt.show()

# Correlation entre sexe et survivre, les femmes avaient plus de chance de survie
# Et les hommes avaient moins de chance de survie :
sns.violinplot(x="Sex", y="Survived", data=titanic, size=6)
plt.show()

sns.boxplot(x="Survived", y="Fare", data=titanic[titanic['Sex']=='female'])
plt.show()

#Les enfants de 17ans de moins avaient plus de chance de survie :
sns.FacetGrid(titanic, hue="Survived", size=5) \
    .map(sns.kdeplot, "Age") \
    .add_legend()
plt.show()

# Visualisation générale
sns.pairplot(titanic,hue="Survived", size=3,diag_kind="kde")
plt.show()

fig = titanic[titanic.Sex=='male'].plot(kind='scatter',x='Age',y='Pclass',color='orange',label='Homme')
titanic[titanic.Sex=='female'].plot(kind='scatter',x='Age',y='Pclass',color='blue',ax=fig,label='femme')
fig.set_xlabel("Age")
fig.set_ylabel("Pclass")
fig.set_title("Age VS Pclass")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

# J'ai mis les 4 modèles pour évaluer la précision.

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split


train, test = train_test_split(titanic, test_size = 0.3)
# print(train.shape)
# print(test.shape)

train_X = train[['Pclass','Survived','Fare']]
train_y=train.Sex
test_X= test[['Pclass','Survived','Fare']] 
test_y =test.Sex  

model = svm.SVC() 
model.fit(train_X,train_y) 
prediction=model.predict(test_X) 
print('La précision du SVM est:',metrics.accuracy_score(prediction,test_y))

model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('La précision de la régression logistique est ',metrics.accuracy_score(prediction,test_y))

model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('La précision de arbre decisionnelle est ',metrics.accuracy_score(prediction,test_y))

model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('La précision de KNN est ',metrics.accuracy_score(prediction,test_y))


