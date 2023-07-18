##################################################
# Imports
##################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    X, y, test_size=0.2, random_state=1039)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=483)
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score Random Forest: ", mean_score)

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=483)
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision Random Forest: ", mean_score)

##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1039)

model = RandomForestClassifier(n_estimators=1276, max_depth=24, min_samples_split=4,
                               min_samples_leaf=4, bootstrap=True, criterion='gini')
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=483)
model = RandomForestClassifier(n_estimators=1276, max_depth=24, min_samples_split=4,
                               min_samples_leaf=4, bootstrap=True, criterion='gini')
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score Random Forest: ", mean_score)

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=483)
model = RandomForestClassifier(n_estimators=1276, max_depth=24, min_samples_split=4,
                               min_samples_leaf=4, bootstrap=True, criterion='gini')
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision Random Forest: ", mean_score)

##################################################

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2)
# clf = RandomForestClassifier()
# param_grid = {'n_estimators': Integer(100, 3000),
#               'max_depth': Integer(1, 200),
#               'min_samples_split': Integer(2, 30),
#               'min_samples_leaf': Integer(1, 4),
#               'bootstrap': Categorical([True, False]),
#               'criterion': Categorical(['gini', 'entropy'])
#               }


# cv = StratifiedKFold(n_splits=3, shuffle=True)

# # The main class from sklearn-genetic-opt
# evolved_estimator = GASearchCV(estimator=clf,
#                                cv=cv,
#                                scoring='accuracy',
#                                param_grid=param_grid,
#                                n_jobs=-1,
#                                verbose=True,
#                                population_size=25,
#                                tournament_size=3,
#                                elitism=True,
#                                generations=50)
# evolved_estimator.fit(X_train, y_train)
# print(evolved_estimator.best_params_)
# y_predict_ga = evolved_estimator.predict(X_test)
# print(accuracy_score(y_test, y_predict_ga))
