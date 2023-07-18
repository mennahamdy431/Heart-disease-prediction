##################################################
# Imports
##################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
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

##################################################

X = heart_important.drop("Heart Disease", axis=1)
y = heart_important["Heart Disease"]

##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=970)

model = XGBClassifier()
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=112)
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score XG: ", mean_score)

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=112)
model = XGBClassifier()
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision XG: ", mean_score)
##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=970)

model = XGBClassifier(learning_rate=0.10169875900115062, max_depth=11, min_child_weight=2,
                      gamma=0.14958327005872096, colsample_bytree=0.691389835271189,
                      n_estimators=95, subsample=0.6336073655520794,
                      colsample_bylevel=0.6115351571764567, colsample_bynode=0.71770121177828)
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=112)
model = XGBClassifier(learning_rate=0.10169875900115062, max_depth=11, min_child_weight=2,
                      gamma=0.14958327005872096, colsample_bytree=0.691389835271189,
                      n_estimators=95, subsample=0.6336073655520794,
                      colsample_bylevel=0.6115351571764567, colsample_bynode=0.71770121177828)
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score XG: ", mean_score)

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=112)
model = XGBClassifier(learning_rate=0.10169875900115062, max_depth=11, min_child_weight=2,
                      gamma=0.14958327005872096, colsample_bytree=0.691389835271189,
                      n_estimators=95, subsample=0.6336073655520794,
                      colsample_bylevel=0.6115351571764567, colsample_bynode=0.71770121177828)
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Precision XG: ", mean_score)

##################################################

# clf = XGBClassifier()
# param_grid = {"learning_rate": Continuous(0.05, 0.30),
#               "max_depth": Integer(3, 15),
#               "min_child_weight": Integer(1, 7),
#               "gamma": Continuous(0.0, 0.5),
#               "colsample_bytree": Continuous(0.3, 1),
#               "n_estimators": Integer(0, 200),
#               "subsample": Continuous(0, 1),
#               "colsample_bylevel": Continuous(0.3, 1),
#               "colsample_bynode": Continuous(0.3, 1)}

##################################################

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
