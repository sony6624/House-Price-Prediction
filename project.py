# ------------------ IMPORT LIBRARIES --------------------
import joblib
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# ------------------ LOAD DATA ---------------------------
data = pd.read_csv("housing.csv")
data.dropna(inplace=True)

print("Initial Data Info\n")
print(data.info())

# ------------------ EDA ---------------------------
# Histogram before transformations
data.hist(figsize=(15, 8))
plt.suptitle("Distribution Before Transformations", fontsize=16)
plt.show()

# Scatter plot of location vs price
plt.figure(figsize=(18,10))
sns.scatterplot(data=data, x='latitude', y='longitude', hue='median_house_value', palette='coolwarm')
plt.title("Housing Prices by Location")
plt.show()

# ------------------ FEATURE ENGINEERING --------------------
data['total_rooms'] = np.log(data['total_rooms'] + 1)
data['total_bedrooms'] = np.log(data['total_bedrooms'] + 1)
data['population'] = np.log(data['population'] + 1)
data['households'] = np.log(data['households'] + 1)

data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
data['household_rooms'] = data['total_rooms'] / data['households']

# One-Hot Encoding
data = data.join(pd.get_dummies(data['ocean_proximity'], dtype=int))
data.drop('ocean_proximity', axis=1, inplace=True)

# ------------------ TRAIN TEST SPLIT --------------------
x = data.drop('median_house_value', axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ------------------ CORRELATION HEATMAP --------------------
corr_matrix = data.corr()
plt.figure(figsize=(22,12))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ------------------ SCALING FOR LINEAR REGRESSION ---------
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

# ------------------ LINEAR REGRESSION ---------------------
lr = LinearRegression()
lr.fit(x_train_s, y_train)

y_pred_lr = lr.predict(x_test_s)

print("\n----- Linear Regression Results -----")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# ------------------ RANDOM FOREST ------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("\n----- Random Forest Results -----")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# ------------------ RANDOMIZED SEARCH TUNING --------------
param_dist = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}

rnd_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=5,   
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42
)

print("\nTuning Random Forest model... please wait...")
rnd_search.fit(x_train, y_train)
best_rf = rnd_search.best_estimator_

joblib.dump(best_rf, "house_price_model.pkl")
print(" Model saved as house_price_model.pkl")

y_pred_best = best_rf.predict(x_test)

print("\n----- Best Tuned Random Forest Results -----")
print("Best Parameters:", rnd_search.best_params_)
print("R2 Score:", r2_score(y_test, y_pred_best))
print("MAE:", mean_absolute_error(y_test, y_pred_best))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))

print("\n Model Training & Evaluation Complete!")












































