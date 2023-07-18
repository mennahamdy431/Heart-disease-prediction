##################################################
# Imports
##################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn import metrics
##################################################

##################################################
heart = pd.read_csv(
    "D:\\Nour's stuff\\Python Course\\Graduation Project\\Heart heart 2.csv")
heart_important = heart.loc[:, ["Slope of ST", "Thallium",
                                "Number of vessels fluro", "FBS over 120", "Chest pain type",
                                "Sex", "Exercise angina", "EKG results",
                                "Heart Disease"]]

heart_important["EKG results"] = (
    heart_important["EKG results"]-heart_important["EKG results"].mean())/heart_important["EKG results"].std()
# print(heart_important.head())

##################################################

X = heart_important.drop("Heart Disease", axis=1)
y = heart_important["Heart Disease"]

##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=228)

model = MLPClassifier()
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

# print(metrics.classification_report(y_test, model.predict(X_test)))

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=440)
model = MLPClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score Neural: ", mean_score)

##################################################

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=440)
model = MLPClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision Neural: ", mean_score)

##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=228)

model = MLPClassifier(activation='relu', solver='adam',
                      alpha=0.030761195377858975, learning_rate='constant', random_state=1522)
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

# print(metrics.classification_report(y_test, model.predict(X_test)))

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=440)
model = MLPClassifier(activation='relu', solver='adam',
                      alpha=0.030761195377858975, learning_rate='constant', random_state=1522)
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score Neural: ", mean_score)

##################################################

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=440)
model = MLPClassifier(activation='relu', solver='adam',
                      alpha=0.030761195377858975, learning_rate='constant', random_state=1522)
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision Neural: ", mean_score)

##################################################

# clf = MLPClassifier(verbose=False)
# param_grid = {
#     'activation': Categorical(['tanh', 'relu']),
#     'solver': Categorical(['sgd', 'adam']),
#     'alpha': Continuous(0.0001, 0.05),
#     'learning_rate': Categorical(['constant', 'adaptive'])
# }

# ##################################################

# cv = StratifiedKFold(n_splits=3, shuffle=True)

# # The main class from sklearn-genetic-opt
# evolved_estimator = GASearchCV(estimator=clf,
#                                cv=cv,
#                                scoring='accuracy',
#                                param_grid=param_grid,
#                                n_jobs=-1,
#                                verbose=True,
#                                population_size=20,
#                                tournament_size=5,
#                                elitism=True,
#                                generations=100)
# evolved_estimator.fit(X_train, y_train)
# print(evolved_estimator.best_params_)
# y_predict_ga = evolved_estimator.predict(X_test)
# print(accuracy_score(y_test, y_predict_ga))

#################################################
