import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# PAGE CONFIG

st.set_page_config(page_title="Student Grade Predictor", layout="centered")
st.title("🎓 Student Final Grade Prediction")
st.markdown("Predict students' final performance (G3) using demographic, social, and academic factors.")

# DATA UPLOAD SECTION

st.header("📤 Upload CSV Dataset")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview:")
    st.dataframe(df.head())

    # FEATURE AND TARGET SELECTION

    target_col = st.selectbox("🎯 Select Target Column (Final Grade)", df.columns, index=len(df.columns)-1)
    feature_cols = st.multiselect(
        "🧩 Select Feature Columns",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col]
    )

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # HANDLE CATEGORICAL DATA

    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # If target is categorical, encode it too
    if y.dtype == "object" or y.dtype.name == "category":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    else:
        le_target = None

    # SPLIT AND SCALE

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MODEL SELECTION

    st.header("🤖 Choose Machine Learning Model")
    model_name = st.selectbox("Select Model", ["Linear Regression", "Logistic Regression", "Random Forest", "XGBoost", "SVM"])

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        model = SVR()


    # TRAIN MODEL

    if st.button("🏋 Train Model"):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        st.success("✅ Model trained successfully!")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Save in session
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["features"] = feature_cols
        st.session_state["encoders"] = label_encoders
        st.session_state["target_encoder"] = le_target

# PREDICT NEW STUDENT

if "model" in st.session_state:
    st.header("🧾 Predict a New Student's Final Grade")

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    features = st.session_state["features"]
    encoders = st.session_state["encoders"]
    le_target = st.session_state["target_encoder"]

    input_data = {}
    for feature in features:
        if feature in encoders:  # categorical
            options = encoders[feature].classes_.tolist()
            val = st.selectbox(f"{feature}", options)
            input_data[feature] = encoders[feature].transform([val])[0]
        else:
            val = st.number_input(f"{feature}", value=0.0)
            input_data[feature] = val

    if st.button("🔮 Predict Grade"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if le_target:
            prediction = le_target.inverse_transform([int(round(prediction))])[0]

        st.success(f"🎓 Predicted Final Grade (G3): {prediction}")