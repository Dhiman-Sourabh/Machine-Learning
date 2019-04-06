import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('data.csv')
dataset = dataset.drop('id', axis = 1)
dataset = dataset.drop('Unnamed: 32', axis = 1)

X = dataset.drop('diagnosis', axis=1).values
y = dataset['diagnosis'].values


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size = .2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

from sklearn.svm import SVC
svc_model = SVC(C = 10, gamma = 0.1, kernel = 'linear' )
svc_model.fit(X_train,Y_train)

Y_pred = svc_model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot = True)

classification_report(Y_test, Y_pred)





