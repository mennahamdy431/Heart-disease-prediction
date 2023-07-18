##################################################
# Imports
##################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1208)
model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score KNN: ", mean_score)

##################################################

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1208)
model = KNeighborsClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision KNN: ", mean_score)

##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=228)

model = KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan',
                             algorithm="brute", leaf_size=57)
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1208)
model = KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan',
                             algorithm="brute", leaf_size=57)
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score KNN: ", mean_score)

##################################################

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1208)
model = KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan',
                             algorithm="brute", leaf_size=57)
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision KNN: ", mean_score)

##################################################

# clf = KNeighborsClassifier()
# param_grid = {'n_neighbors': Integer(3, 15),
#               'weights': Categorical(['uniform', 'distance']),
#               'metric': Categorical(['minkowski', 'euclidean', 'manhattan']),
#               "algorithm": Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
#               "leaf_size": Integer(2, 100)
#               }


# cv = StratifiedKFold(n_splits=5, shuffle=True)

# The main class from sklearn-genetic-opt
# evolved_estimator = GASearchCV(estimator=clf,
#                                cv=cv,
#                                scoring='accuracy',
#                                param_grid=param_grid,
#                                n_jobs=-1,
#                                verbose=True,
#                                population_size=100,
#                                tournament_size=5,
#                                elitism=True,
#                                generations=250)
# evolved_estimator.fit(X_train, y_train)
# print(evolved_estimator.best_params_)
# y_predict_ga = evolved_estimator.predict(X_test)
# print(accuracy_score(y_test, y_predict_ga))
