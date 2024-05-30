import hw5

import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, median_absolute_error, explained_variance_score, max_error
from scipy.stats import pearsonr, spearmanr

import json

def open_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
            print("JSON is valid.")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
    return data


# train_df = hw5.read_json("data/rtvslo_train.json.gz")
# test_df = hw5.read_json("data/rtvslo_test.json.gz")
train_df = open_json("data/rtvslo_train.json")
test_df = open_json("data/rtvslo_test.json")


instanca = hw5.RTVSlo()
print("-" * 100 + "\nPerforming fit...\n" + "-" * 100)
instanca.fit(train_df)
print("-" * 100 + "\nPerforming predict...\n" + "-" * 100)
y_pred = instanca.predict(test_df)
print("-" * 100 + "\nAll done...\n" + "-" * 100)
print(y_pred, "\n")

# y_test = pd.DataFrame(test_df)["n_comments"]

# mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)

# try:
#     msle = mean_squared_log_error(y_test, y_pred)
# except:
#     msle = float("nan")
# medae = median_absolute_error(y_test, y_pred)
# evs = explained_variance_score(y_test, y_pred)
# max_err = max_error(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"R-squared (R²): {r2}")
# print()
# print(f"Mean Squared Logarithmic Error (MSLE): {msle}")
# print(f"Median Absolute Error: {medae}")
# print(f"Explained Variance Score: {evs}")
# print(f"Max Error: {max_err}")

# pearson_corr, _ = pearsonr(y_test, y_pred)
# spearman_corr, _ = spearmanr(y_test, y_pred)

# print(f"Pearson Correlation Coefficient: {pearson_corr}")
# print(f"Spearman Rank Correlation Coefficient: {spearman_corr}")

print("Coefficients:", instanca.model.coef_)
print("Intercept:", instanca.model.intercept_)

np.savetxt("prev_ver/results/predictions.txt", y_pred, fmt="%f")