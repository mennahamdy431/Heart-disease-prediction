import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
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

##################################################

X = heart_important.drop("Heart Disease", axis=1)
y = heart_important["Heart Disease"]
#######################################


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1039)

xg_classifier = XGBClassifier(learning_rate=0.10169875900115062, max_depth=11, min_child_weight=2,
                              gamma=0.14958327005872096, colsample_bytree=0.691389835271189,
                              n_estimators=95, subsample=0.6336073655520794,
                              colsample_bylevel=0.6115351571764567, colsample_bynode=0.71770121177828)
cat_classifier = CatBoostClassifier(verbose=False, learning_rate=0.011906604774021327, depth=7,
                                    l2_leaf_reg=5, model_size_reg=3.222228730698581, n_estimators=382,
                                    random_strength=2.8154106630677678)
knn_classifier = KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan',
                                      algorithm="brute", leaf_size=57)
model = VotingClassifier(
    estimators=[("xg", xg_classifier), ("cat", cat_classifier),
                ("knn", knn_classifier)], voting="soft")

model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1093)
model = VotingClassifier(
    estimators=[("xg", xg_classifier), ("cat", cat_classifier),
                ("knn", knn_classifier)], voting="soft")
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score Voting: ", mean_score)

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1093)
model = VotingClassifier(
    estimators=[("xg", xg_classifier), ("cat", cat_classifier),
                ("knn", knn_classifier)], voting="soft")
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Score Voting: ", mean_score)

##################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=269)

xg_classifier = XGBClassifier(learning_rate=0.10169875900115062, max_depth=11, min_child_weight=2,
                              gamma=0.14958327005872096, colsample_bytree=0.691389835271189,
                              n_estimators=95, subsample=0.6336073655520794,
                              colsample_bylevel=0.6115351571764567, colsample_bynode=0.71770121177828)
cat_classifier = CatBoostClassifier(verbose=False, learning_rate=0.011906604774021327, depth=7,
                                    l2_leaf_reg=5, model_size_reg=3.222228730698581, n_estimators=382,
                                    random_strength=2.8154106630677678)
knn_classifier = KNeighborsClassifier(n_neighbors=6, weights='uniform', metric='manhattan',
                                      algorithm="brute", leaf_size=57)

nn_classifier = MLPClassifier(activation='relu', solver='adam',
                              alpha=0.030761195377858975, learning_rate='constant', random_state=1522)


rf_classifier = RandomForestClassifier(n_estimators=1276, max_depth=24, min_samples_split=4,
                                       min_samples_leaf=4, bootstrap=True, criterion='gini')


model = VotingClassifier(
    estimators=[("xg", xg_classifier), ("cat", cat_classifier),
                ("knn", knn_classifier), ("NN", nn_classifier), ("RF", rf_classifier)],
    voting="soft")
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=269)
model = VotingClassifier(
    estimators=[("xg", xg_classifier), ("cat", cat_classifier),
                ("knn", knn_classifier), ("NN", nn_classifier), ("RF", rf_classifier)],
    voting="soft")
scores = cross_val_score(model, X, y, cv=sk_folds)
mean_score = round(scores.mean(), 3)
print("Average CV Score Voting: ", mean_score)

##################################################

sk_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=269)
model = VotingClassifier(
    estimators=[("xg", xg_classifier), ("cat", cat_classifier),
                ("knn", knn_classifier), ("NN", nn_classifier), ("RF", rf_classifier)],
    voting="soft")
scores = cross_val_score(model, X, y, cv=sk_folds, scoring="precision")
mean_score = round(scores.mean(), 3)
print("Average CV Score Voting: ", mean_score)

##################################################
