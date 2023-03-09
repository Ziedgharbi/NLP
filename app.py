import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import glob


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


directory_path="C:/Users/pc/Nextcloud/Python/GITHUB/NLP/"
data_path=directory_path+"data/"



file_name=[ name.split("\\",1)[1] for name in  glob.glob(data_path+"*.csv")  ]

df=pd.DataFrame()

for name in file_name:
    df_temp=pd.read_csv(data_path+name)
    df=pd.concat([df,df_temp])
    
df.info()
df.CLASS.value_counts(normalize=True)
df["CLASS"].value_counts().plot.bar()
df.CLASS.value_counts(normalize=True).plot.bar()
df["CLASS"].value_counts().plot.pie()
df.CLASS.value_counts(normalize=True).plot.pie()

#CLASS=1 ---> Spam  //  CLASS=0 ----> non SPAM

df["CONTENT"].iloc[3]


## split data 
seed=4040

X=df.CONTENT
y=df.CLASS

x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, shuffle=True, random_state=seed)


# vectorize text feature 
vectorizer=CountVectorizer()
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)

vectorizer.get_feature_names_out() # see content of features


# apply random forest classifier

randf=RandomForestClassifier(random_state=seed)
rst=randf.fit(x_train_vec, y_train)

# score on test set
rst.score(x_test_vec, y_test)

### classification repot
y_pred=rst.predict(x_test_vec)
report=classification_report(y_test, y_pred)
print(report)

## confusion matrix

conf_matrix=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()



## Testing performence by using cross validation : mean of performence
scores=cross_val_score(randf, x_train_vec,y_train,cv=4)

scores.mean()



## test prediction

rst.predict(vectorizer.transform (['i enjoyed a lot']))
rst.predict(vectorizer.transform (['come and enjoy me on xxx fun']))



