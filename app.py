# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load data and model
# df = pd.read_csv("currency_data.csv")

# # Rename column if needed (fix lowercase date column)
# if 'date' in df.columns and 'Date' not in df.columns:
#     df.rename(columns={'date': 'Date'}, inplace=True)

# # Convert 'Date' column to datetime if it exists
# if 'Date' in df.columns:
#     df['Date'] = pd.to_datetime(df['Date'])

# # Load the trained model
# model = joblib.load("forex_model.pkl")

# st.set_page_config(page_title="Currency Rate Predictor", layout="centered")

# # App Title
# st.title("💹 Currency Rate Movement Classifier")

# # Sidebar: User Inputs
# st.sidebar.header("🔢 Enter Currency Features")

# open_val = st.sidebar.number_input("Open Price", min_value=0.0, format="%.4f")
# high_val = st.sidebar.number_input("High Price", min_value=open_val, format="%.4f")
# low_val = st.sidebar.number_input("Low Price", max_value=high_val, format="%.4f")
# volume_val = st.sidebar.number_input("Volume", min_value=0)

# # Predict button
# if st.sidebar.button("Predict Movement"):
#     input_data = pd.DataFrame([{
#         "Open": open_val,
#         "High": high_val,
#         "Low": low_val,
#         "Volume": volume_val
#     }])

#     prediction = model.predict(input_data)[0]
#     prediction_label = "📈 Increase Expected" if prediction == 1 else "📉 Decrease Expected"

#     st.subheader("📌 Prediction Result")
#     st.success(prediction_label)

# # Visualization Section
# st.subheader("📊 Historical Data Visualization")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("Closing Price Trend")

#     if 'Date' in df.columns and 'Close' in df.columns:
#         fig1, ax1 = plt.subplots()
#         sns.lineplot(data=df.tail(100), x="Date", y="Close", ax=ax1)
#         plt.xticks(rotation=45)
#         st.pyplot(fig1)
#     else:
#         st.warning("Date or Close column missing in data.")

# with col2:
#     st.markdown("Volume Distribution")

#     if 'Volume' in df.columns:
#         fig2, ax2 = plt.subplots()
#         sns.histplot(data=df, x="Volume", bins=20, ax=ax2, color="skyblue")
#         st.pyplot(fig2)
#     else:
#         st.warning("Volume column missing in data.")

# # Show raw data
# st.subheader("📄 Raw Data Preview")
# st.dataframe(df.tail(10))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and model
df = pd.read_csv("currency_data.csv")

# Fix lowercase date column if needed
if 'date' in df.columns and 'Date' not in df.columns:
    df.rename(columns={'date': 'Date'}, inplace=True)

# Convert 'Date' column to datetime if it exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Load the trained model
model = joblib.load("forex_model.pkl")

# Page config
st.set_page_config(page_title="Currency Rate Predictor", layout="centered")

# Title
st.title("💹 Currency Rate Movement Classifier")

# Sidebar Inputs
st.sidebar.header("🔢 Enter EUR/USD Features")

open_val = st.sidebar.number_input("Open Price", min_value=0.0, format="%.4f")
high_val = st.sidebar.number_input("High Price", min_value=open_val, format="%.4f")
low_val = st.sidebar.number_input("Low Price", max_value=high_val, format="%.4f")
volume_val = st.sidebar.number_input("Tick Volume", min_value=0)

# Predict
if st.sidebar.button("Predict Movement"):
    input_data = pd.DataFrame([{
        "open_eurusd": open_val,
        "high_eurusd": high_val,
        "low_eurusd": low_val,
        "tikvol_eurusd": volume_val
    }])

    prediction = model.predict(input_data)[0]
    prediction_label = "📈 Increase Expected" if prediction == 1 else "📉 Decrease Expected"

    st.subheader("📌 Prediction Result")
    st.success(prediction_label)

# Visualizations
st.subheader("📊 Historical EUR/USD Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("📈 Closing Price Trend (EUR/USD)")

    if 'Date' in df.columns and 'close_eurusd' in df.columns:
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=df.tail(100), x="Date", y="close_eurusd", ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    else:
        st.warning("Missing Date or close_eurusd column.")

with col2:
    st.markdown("📊 Tick Volume Distribution")

    if 'tikvol_eurusd' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x="tikvol_eurusd", bins=20, ax=ax2, color="skyblue")
        st.pyplot(fig2)
    else:
        st.warning("Missing tikvol_eurusd column.")

# Show raw data
st.subheader("📄 Raw Data Preview")
st.dataframe(df.tail(10))
