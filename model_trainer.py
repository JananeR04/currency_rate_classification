import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("currency_data.csv")

# Use EUR/USD features only
df = df[["open_eurusd", "high_eurusd", "low_eurusd", "close_eurusd", "tikvol_eurusd"]].copy()

# Create binary target: 1 if next hour's close > current open
df["target"] = (df["close_eurusd"].shift(-1) > df["open_eurusd"]).astype(int)
df.dropna(inplace=True)

# Features and target
X = df[["open_eurusd", "high_eurusd", "low_eurusd", "tikvol_eurusd"]]
y = df["target"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "forex_model.pkl")
print("✅ Model trained and saved as forex_model.pkl")
