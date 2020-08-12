# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 23:43:44 2020

@author: Rohith
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import wordcloud
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

# Loading dataset

data= pd.read_csv('E:\\Data Science\\Data_Science_Projects\\Project_7_Review Analysis for A product Listed on Amazon\\Dataset\\data.csv') 

data.head(10)
data.columns
data.shape
data.info()
data.describe()
data.dtypes



#################################Exploratory Data Analysis######################################

ratings=data.groupby(['Rating']).count 
ratings

###finding Number of unique values
data.nunique()
##Checking Null values
data.isnull().sum()
# Dataset in Graphs

plt.figure(figsize=(12,8))
data['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Count')


# Plot distribution of review length
review_length = data["Reviews"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12,8))
review_length.loc[review_length < 1500].hist()
plt.title("Distribution of Review Length")
plt.xlabel('Review length (Number of character)')
plt.ylabel('Count')

data.Rating.hist()
data.Rating.hist(bins=10)
plt.xlabel('Rating')
plt.ylabel('Count')

##considering Review column
Reviews=data.iloc[:,[4]] Reviews.shape
Reviews.describe()
Reviews.head(10)
Reviews.dtypes

## FINDING NULL VALUES...##
data.isnull().sum() ## there are no null values #

## droping all NA values from Train  data..##
data.dropna(inplace=True)

#############......Pre-processing on train data...###########################


# removing the date column as date has not that significance in output##
data.drop(["Customer Name","Review Title"],axis=1,inplace=True)

data.head()

## cleaning the data..##
## Cleaning the text input for betting understanding of Machine..##

##Converting all review into Lowercase..###
data['Reviews']= data['Reviews'].apply(lambda x: " ".join(word.lower() for word in x.split()))

## removing punctuation from review..#
import string
data['Reviews']=data['Reviews'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))
                                                 

## Remove Numbers from review...##
data['Reviews']=data['Reviews'].str.replace('[0-9]','')


## removing all stopwords(english)....###
from nltk.corpus import stopwords

stop_words=stopwords.words('english')

data['Reviews']=data['Reviews'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

data.head(2)

# Lemmatization
from textblob import Word
data['Reviews']= data['Reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


import re
pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
data['Reviews']= data['Reviews'].apply(lambda x:(re.sub(pattern, '',x).strip()))
data['Reviews'].head()


Review_wordcloud = ' '.join(data['Reviews'])
Q_wordcloud=WordCloud(
                    background_color='white',
                    width=2000,
                    height=2000
                   ).generate(Review_wordcloud)
fig = plt.figure(figsize = (10, 10))
plt.axis('on')
plt.imshow(Q_wordcloud)

###top 20 most frequent repeated words from Reviews
freq = pd.Series(' '.join(data['Reviews']).split()).value_counts()[0:20]
freq


################# bigrams for Reviews #####################
import collections
from collections import Counter

counts_Reviews = collections.Counter()
for i in data['Reviews']:
    words_A = word_tokenize(i)
    counts_Reviews.update(nltk.bigrams(words_A))    
Bigram_counts_Reviews = counts_Reviews.most_common(10)
Bigram_counts_Reviews

############ TFIDF matrix ################
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
TFIDF=tfidf.fit_transform(data['Reviews'])
print(TFIDF)


################## Sentiment Analysis #####################3
from textblob import TextBlob
data['polarity'] = data['Reviews'].apply(lambda x: TextBlob(x).sentiment[0])
data[['Reviews','polarity']].head(5)

# Displaying top 5 positive posts of Category_A
data[data.polarity>0].head(5)



# ======= The distribution of Categories polarity score =======
    
sns.set()
plt.hist(x='polarity', data=data, bins=20);
plt.xlabel('polarity of Reviews');
plt.ylabel('count'); 
plt.figsize=(10, 16)



def sent_type(text): 
    for i in (text):
        if i>0:
            print('positive')
        elif i==0:
            print('netural')
        else:
            print('negative') 

sent_type(data['polarity'])

data["category"]=data['polarity']

data.loc[data.category>0,'category']="Positive"
data.loc[data.category!='Positive','category']="Negative"

data["category"]=data["category"].astype('category')
data.dtypes

data["category"].value_counts()

sns.countplot(x='category',data=data,palette='hls')

#############....Positive & Negative reviews WordCloud formation ################################

positive_reviews= data[data.category=='Positive']
negative_reviews= data[data.category=='Negative']


positive_reviews_text=" ".join(positive_reviews.Reviews.to_numpy().tolist())
negative_reviews_text=" ".join(negative_reviews.Reviews.to_numpy().tolist())

positive_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(positive_reviews_text)
negative_reviews_cloud=WordCloud(background_color='black',max_words=150).generate(negative_reviews_text)

plt.imshow(positive_reviews_cloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()

plt.imshow(negative_reviews_cloud,interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0) 
plt.show()

####################################################################################################################################

###########Model_Building####################################
 ###Logistic_Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

tv=TfidfVectorizer()

X=data.iloc[:,2]
y=data.iloc[:,4]

X=tv.fit_transform(data.Reviews)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

n_errors_Log=print((y_pred!=y_test).sum())
cohen_kappa_score(y_test,y_pred)
print('Training accuracy:',accuracy_score(y_train,model.predict(X_train)))


########################Decision tree Classifier#######################################################print(accuracy_score(y_train,model.predict(X_train)))

from sklearn.tree import DecisionTreeClassifier

model1=DecisionTreeClassifier(criterion='entropy')
model1.fit(X_train,y_train)

y_pred1=model1.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))

n_errors_Dec=print((y_pred1!=y_test).sum())
cohen_kappa_score(y_test,y_pred1)
print('Training accuracy:',accuracy_score(y_train,model1.predict(X_train)))

#############################Random Forest Classifier#######################################################

from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(n_estimators=100)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Testing accuracy:',accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

n_errors_Ran=print((y_pred2!=y_test).sum())
cohen_kappa_score(y_test,y_pred2)
print('Training accuracy:',accuracy_score(y_train,model2.predict(X_train)))

###################################Extratreeclassifier####################################################

from sklearn.ensemble import ExtraTreesClassifier 

model3=ExtraTreesClassifier()
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
n_errors_ext=print((y_pred3!=y_test).sum())

cohen_kappa_score(y_test,y_pred3) 
print('Training accuracy:',accuracy_score(y_train,model3.predict(X_train)))

####################################Support Vector Machine ####################################################################################

from sklearn.svm import SVC  

model4=SVC()
model4.fit(X_train,y_train)

y_pred4=model4.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred4))
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))

n_errors_svc=print((y_pred4!=y_test).sum())
cohen_kappa_score(y_test,y_pred4)
print('Training accuracy:',accuracy_score(y_train,model4.predict(X_train)))

#################################Neural_Networks###############################################################################################################

from sklearn.neural_network import MLPClassifier 

model5=MLPClassifier(hidden_layer_sizes=(5,5))
model5.fit(X_train,y_train)
y_pred5=model5.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test,y_pred5))
print(classification_report(y_test,y_pred5))

n_errors_nn=print((y_pred5!=y_test).sum())
cohen_kappa_score(y_test,y_pred5)
print('Training accuracy:',accuracy_score(y_train,model5.predict(X_train)))

##############################Bagging Classifier##################################################################################################################


from sklearn.ensemble import BaggingClassifier 

model6=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model6.fit(X_train,y_train)

y_pred6=model6.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred6))
print(confusion_matrix(y_test,y_pred6))
print(classification_report(y_test,y_pred6))


n_errors_BC=print((y_pred6!=y_test).sum())
cohen_kappa_score(y_test,y_pred6) 
print('Training accuracy:',accuracy_score(y_train,model6.predict(X_train)))



###############################Extreme Grdient Boosting Algorithm ########################################

import xgboost as xgb 

model7=xgb.XGBClassifier()
model7.fit(X_train,y_train)

y_pred7=model7.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred7))
print(confusion_matrix(y_test,y_pred7))
print(classification_report(y_test,y_pred7))


n_errors_BC=print((y_pred7!=y_test).sum())
cohen_kappa_score(y_test,y_pred7) 
print('Training accuracy:',accuracy_score(y_train,model7.predict(X_train)))


#.We can observe majority class for no class and monority class for yes class
#So we need to solve class imbalance problem with the help of SMOTE

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)

X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape

########################LogisticRegression after SMOTE#######################################################print(accuracy_score(y_train,model.predict(X_train)))


from sklearn.linear_model import LogisticRegression 

model9=LogisticRegression()
model9.fit(X_train_res,y_train_res)
y_pred9=model9.predict(X_test)
 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred9))
print(confusion_matrix(y_test,y_pred9))
print(classification_report(y_test,y_pred9))
n_errors_Log_SM=print((y_pred9!=y_test).sum())
cohen_kappa_score(y_test,y_pred9)
print('Training accuracy:',accuracy_score(y_train,model9.predict(X_train)))


########################Decision tree Classifier#######################################################print(accuracy_score(y_train,model.predict(X_train)))

from sklearn.tree import DecisionTreeClassifier

model10=DecisionTreeClassifier(criterion='entropy')
model10.fit(X_train_res,y_train_res)

y_pred10=model10.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred10))
print(confusion_matrix(y_test,y_pred10))
print(classification_report(y_test,y_pred10))

n_errors_Dec_SM=print((y_pred10!=y_test).sum())
cohen_kappa_score(y_test,y_pred10)
print('Training accuracy:',accuracy_score(y_train,model10.predict(X_train)))

#############################Random Forest Classifier#######################################################

from sklearn.ensemble import RandomForestClassifier
model11=RandomForestClassifier(n_estimators=100)
model11.fit(X_train_res,y_train_res)
y_pred11=model11.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Testing accuracy:',accuracy_score(y_test,y_pred11))
print(confusion_matrix(y_test,y_pred11))
print(classification_report(y_test,y_pred11))

n_errors_Ran_SM=print((y_pred11!=y_test).sum())
cohen_kappa_score(y_test,y_pred11)
print('Training accuracy:',accuracy_score(y_train,model11.predict(X_train)))

###################################Extratreeclassifier####################################################

from sklearn.ensemble import ExtraTreesClassifier 

model12=ExtraTreesClassifier()
model12.fit(X_train_res,y_train_res)
y_pred12=model2.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('Testing accuracy:',accuracy_score(y_test,y_pred12))
print(confusion_matrix(y_test,y_pred12))
print(classification_report(y_test,y_pred12))
n_errors_ext_SM=print((y_pred12!=y_test).sum())

cohen_kappa_score(y_test,y_pred12) 
print('Training accuracy:',accuracy_score(y_train,model12.predict(X_train)))

####################################Support Vector Machine ####################################################################################

from sklearn.svm import SVC  

model13=SVC()
model13.fit(X_train_res,y_train_res)

y_pred13=model13.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred13))
print(confusion_matrix(y_test,y_pred13))
print(classification_report(y_test,y_pred13))

n_errors_svc_SM=print((y_pred13!=y_test).sum())
cohen_kappa_score(y_test,y_pred13)
print('Training accuracy:',accuracy_score(y_train,model13.predict(X_train)))

#################################Neural_Networks###############################################################################################################

from sklearn.neural_network import MLPClassifier 

model14=MLPClassifier(hidden_layer_sizes=(5,5))
model14.fit(X_train_res,y_train_res)
y_pred14=model14.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred14))
print(confusion_matrix(y_test,y_pred14))
print(classification_report(y_test,y_pred14))

n_errors_nn_SM=print((y_pred14!=y_test).sum())
cohen_kappa_score(y_test,y_pred14)
print('Training accuracy:',accuracy_score(y_train,model14.predict(X_train)))

##############################Bagging Classifier##################################################################################################################


from sklearn.ensemble import BaggingClassifier 

model15=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model15.fit(X_train_res,y_train_res)

y_pred15=model15.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred15))
print(confusion_matrix(y_test,y_pred15))
print(classification_report(y_test,y_pred15))


n_errors_BC_SM=print((y_pred15!=y_test).sum())
cohen_kappa_score(y_test,y_pred15) 
print('Training accuracy:',accuracy_score(y_train,model15.predict(X_train)))



###############################Extreme Grdient Boosting Algorithm ########################################

import xgboost as xgb 

model16=xgb.XGBClassifier()
model16.fit(X_train_res,y_train_res)

y_pred16=model16.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred16))
print(confusion_matrix(y_test,y_pred16))
print(classification_report(y_test,y_pred16))


n_errors_BC_SM=print((y_pred16!=y_test).sum())
cohen_kappa_score(y_test,y_pred16) #0.5220
print('Training accuracy:',accuracy_score(y_train,model16.predict(X_train)))


####################################Linear SVC##################################################
from sklearn.svm import LinearSVC

model18= LinearSVC()

model18.fit(X_train,y_train)

y_pred18=model18.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred18))
print(confusion_matrix(y_test,y_pred18))
print(classification_report(y_test,y_pred18))


n_errors_LSVC=print((y_pred18!=y_test).sum())
cohen_kappa_score(y_test,y_pred18) 
print('Training accuracy:',accuracy_score(y_train,model18.predict(X_train)))

##################################Linear SVC _after smote##########################################

from sklearn.svm import LinearSVC

model19= LinearSVC()

model19.fit(X_train_res,y_train_res)

y_pred19=model19.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred19))
print(confusion_matrix(y_test,y_pred19))
print(classification_report(y_test,y_pred19))


n_errors_LSVCSM=print((y_pred19!=y_test).sum())
cohen_kappa_score(y_test,y_pred19) 
print('Training accuracy:',accuracy_score(y_train,model19.predict(X_train)))

####################################Naive Bayes_before SMOTE##################################################
from sklearn.naive_bayes import MultinomialNB

model20= MultinomialNB()

model20.fit(X_train,y_train)

y_pred20=model20.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred20))
print(confusion_matrix(y_test,y_pred20))
print(classification_report(y_test,y_pred20))


n_errors_MNB=print((y_pred20!=y_test).sum())
cohen_kappa_score(y_test,y_pred20) 
print('Training accuracy:',accuracy_score(y_train,model20.predict(X_train)))

##################################Naive Bayes _after smote##########################################

from sklearn.naive_bayes import MultinomialNB

model21= MultinomialNB()

model21.fit(X_train_res,y_train_res)

y_pred21=model21.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print('Testing accuracy:',accuracy_score(y_test,y_pred21))
print(confusion_matrix(y_test,y_pred21))
print(classification_report(y_test,y_pred21))


n_errors_MNBSM=print((y_pred21!=y_test).sum())
cohen_kappa_score(y_test,y_pred21) 
print('Training accuracy:',accuracy_score(y_train,model21.predict(X_train)))
#####################################################################################################
