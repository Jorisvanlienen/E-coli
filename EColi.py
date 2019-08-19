from typing import List, Any
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
names = ['Class', 'id', 'Sequence']
data = pd.read_csv(url, names = names)

classes = data.loc[:, 'Class']
print(classes[:5])

sequences = list(data.loc[:, 'Sequence'])
dataset = {}

for i, seq in enumerate(sequences):
    nucleotides = List[Any] = list(seq)
    nucleotides = [x for x in nucleotides if x != '\t']
    nucleotides.append(classes[i])
    dataset[i] = nucleotides
print(dataset[0])

dframe = pd.DataFrame(dataset)
print(dframe)
df = dframe.transpose()
print(df.iloc[:5])

df.rename(columns = {57: 'Class'}, inplace = True)
print(df.iloc[:5])

series = []
for name in df.columns:
 series.append(df[name].value_counts())

info = pd.DataFrame(series)
details = info.transpose()
print(details)

numerical_df = pd.get_dummies(df)
numerical_df.iloc[:5]

df = numerical_df.drop(columns=['Class_-'])
df.rename(columns = {'Class_+': 'Class'}, inplace = True)
print(df.iloc[:5])

df = numerical_df.drop(columns=['Class_-'])

df.rename(columns = {'Class_+': 'Class'}, inplace = True)
print(df.iloc[60])

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])
seed = 1

# split train test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=seed)

scoring = 'accuracy'

# Classifiers
names = ["Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "SVM Linear", "SVM RBF", "SVM Sigmoid"]

classifiers = [
    KNeighborsClassifier(n_neighbors = 3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10,
    max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel = 'linear'),
    SVC(kernel = 'rbf'),
    SVC(kernel = 'sigmoid')
]
models = zip(names, classifiers)

# evaluation classifiers
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))