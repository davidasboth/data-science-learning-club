import pandas as pd
import numpy as np

from gaussian_nb import GaussianNaiveBayes
from evaluation import Evaluator

# Titanic dataset
df = pd.read_csv("data/titanic/train.csv")
# extract the relevant columns
df_cols = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived']]
df_cols.dropna(axis=0, inplace=True)

# turn categorical features into numbers
gender = list(np.unique(df_cols['Sex'].values))
df_cols['Sex'] = df_cols['Sex'].apply(lambda s: gender.index(s))

embarked = list(np.unique(df_cols['Embarked'].values))
df_cols['Embarked'] = df_cols['Embarked'].apply(lambda e: embarked.index(e))

# extract features and targets into 2 different variables
targets = df_cols['Survived']
features = df_cols.drop(['Survived'], axis=1)

# extract validation sets
x_train = features.iloc[:500,:]
y_train = targets[:500]

x_test = features.iloc[500:,:]
y_test = targets[500:]

nb = GaussianNaiveBayes(x_train,
                        y_train,
                        categoricals=[True, True, False, True, True, True],
                        debug_mode=False)

nb.train()

predictions = nb.predict(x_test)

ev = Evaluator()
metrics = ev.evaluate(predictions, y_test, binary=True)
print("Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F-Score: {:.3f}".format(metrics.accuracy,
                                                                    metrics.precision,
                                                                    metrics.recall,
                                                                    metrics.f_score))

# for benchmarking, try sklearn implementation
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as sk_metrics

gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_predictions = gnb.predict(x_test)

print("\nSKLearn accuracy: {:.3f}".format(sk_metrics.accuracy_score(y_test, gnb_predictions)))