import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cat = {"Model": "CatBoost", "Accuracy": [
    0.845, 0.864], "Optimized": [False, True]}
forest = {"Model": "Random Forest", "Accuracy": [
    0.827, 0.853], "Optimized": [False, True]}
knn = {"Model": "K Neighbors", "Accuracy": [
    0.819, 0.846], "Optimized": [False, True]}
nn = {"Model": "Neural Network", "Accuracy": [
    0.792, 0.817], "Optimized": [False, True]}
svm = {"Model": "Support Vector Machine", "Accuracy": [
    0.82, 0.836], "Optimized": [False, True]}
xg = {"Model": "XGBoost", "Accuracy": [
    0.803, 0.859], "Optimized": [False, True]}

models = [cat, forest, knn, nn, svm, xg]
df = pd.DataFrame(models)

# Create the final DataFrame
final_df = pd.DataFrame({
    "Model": df["Model"].repeat(2),
    "Accuracy": [acc for acc_list in df["Accuracy"] for acc in acc_list],
    "Optimized": [opt for opt_list in df["Optimized"] for opt in opt_list]
})

# Reset the index
final_df.reset_index(drop=True, inplace=True)

print(final_df)
##################################################
sns.set_palette("Spectral", n_colors=6, desat=0.8)
ax = sns.barplot(data=final_df, x="Accuracy", y="Model", hue="Optimized")
plt.title("Cross Validation Accuracy Scores with different Hyperparameters",
          fontweight="bold", fontsize="26")
ax.bar_label(ax.containers[0], fontweight="bold")
ax.bar_label(ax.containers[1], fontweight="bold")
plt.xlabel("Accuracy score", fontweight="bold", fontsize="large")
plt.ylabel("Model", fontweight="bold", fontsize="large")
plt.show()
