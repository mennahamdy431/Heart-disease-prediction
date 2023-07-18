from joblib import Parallel, delayed
import joblib

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
   "E:\\FCAI\\graduation project\\prooooject\\Heart heart 2.csv")
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

model = RandomForestClassifier(n_estimators=1276, max_depth=24, min_samples_split=4,
                               min_samples_leaf=4, bootstrap=True, criterion='gini')
model.fit(X_train, y_train)
print(round(model.score(X_test, y_test), 3))

# print(metrics.classification_report(y_test, model.predict(X_test)))

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[True, False])

cm_display.plot(cmap=plt.cm.Reds)
plt.show()



##################################################

##################################################

joblib.dump(model, 'rand_model.pkl')
