import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('uber.csv')
print("\n--- Starting Data Preprocessing ---")

if 'key' in df.columns:
    df = df.drop('key', axis=1)

print(f"Missing values before cleaning:\n{df.isnull().sum()}")
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
# df.dropna(subset=['pickup_datetime'], inplace=True)

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour


def manhattan_distance(lat1,lon1,lat2,lon2):
    return abs(lat1 - lat2) + abs(lon1 - lon2)

df['distance_km'] = manhattan_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

df = df.drop([
    'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude'
], axis=1)

print("\nData preprocessing complete.")

print("\n--- Visualizing distributions BEFORE removing outliers ---")
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); sns.boxplot(x=df['fare_amount']); plt.title('Fare Amount (Before)')
plt.subplot(1,3,2); sns.boxplot(x=df['distance_km']); plt.title('Distance (Before)')
plt.subplot(1,3,3); sns.boxplot(x=df['passenger_count']); plt.title('Passenger Count (Before)')
plt.tight_layout(); plt.show()

print("\n--- Removing Outliers using IQR ---")
def remove_outliers_iqr(df, column, min_value=None):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if min_value is not None:
        lower = max(lower, min_value)
    print(f"{column}: removing outside [{lower:.2f}, {upper:.2f}]")
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers_iqr(df, 'fare_amount', min_value=0)
df = remove_outliers_iqr(df, 'distance_km', min_value=0)
df = remove_outliers_iqr(df, 'passenger_count', min_value=1)

df['passenger_count'] = df['passenger_count'].astype(int)

print(f"Dataset shape after outlier removal: {df.shape}")

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); sns.boxplot(x=df['fare_amount']); plt.title('Fare Amount (After)')
plt.subplot(1,3,2); sns.boxplot(x=df['distance_km']); plt.title('Distance (After)')
plt.subplot(1,3,3); sns.boxplot(x=df['passenger_count']); plt.title('Passenger Count (After)')
plt.tight_layout(); plt.show()

print("\n--- Checking Correlation ---")
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

features = ['passenger_count','distance_km','year','month','day','day_of_week','hour']
target = 'fare_amount'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

linear_model = LinearRegression().fit(X_train, y_train)
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)

print("Model trained")

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{name}: R2={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
    return y_pred, r2, mae, rmse

y_pred_lr, lr_r2, lr_mae, lr_rmse = evaluate_model(linear_model, X_test, y_test, "Linear Regression")
y_pred_rf, rf_r2, rf_mae, rf_rmse = evaluate_model(random_forest_model, X_test, y_test, "Random Forest")

print("\n--- Model Comparison ---")
if rf_rmse < lr_rmse:
    print("Random Forest performs better (lower RMSE).")
else:
    print("Linear Regression performs better (lower RMSE).")

# --- Scatter plot: Actual vs Predicted ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Linear Regression", color='blue')
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest", color='green')
plt.plot([0,25],[0,25],'k--', label="Perfect Fit")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.legend()
plt.title("Actual vs Predicted Fares")
plt.show()