import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

st.set_page_config(page_title="Low-Code ML Platform", layout="centered")
st.title("🧠 Low-Code ML Platform")

uploaded_file = st.file_uploader("📁 Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Dataset", df.head())

    task = st.selectbox("🎯 Select Machine Learning Task", ["Classification", "Regression", "Clustering"])

    if task != "Clustering":
        target_column = st.selectbox("🧬 Select Target Column", df.columns)
    else:
        target_column = None

    if st.button("🚀 Train Model"):
        try:
            st.info("Preparing your model...")

            if task == "Clustering":
                X = df.select_dtypes(include=['number'])
                model = KMeans(n_clusters=3, random_state=0)
                model.fit(X)
                df["Cluster"] = model.labels_
                st.success("✅ Clustering completed.")
                st.write(df.head())
            else:
                X = df.drop(columns=[target_column])
                y = df[target_column]

                if task == "Classification" and y.dtype == 'object':
                    y = y.astype('category').cat.codes

                X = pd.get_dummies(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                if task == "Classification":
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Classification model trained! Accuracy: {acc:.2%}")
                elif task == "Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    st.success(f"✅ Regression model trained! MSE: {mse:.4f}")

                model_features = list(X.columns)
                with open("model.pkl", "wb") as f:
                    pickle.dump((model, model_features), f)

                with open("model.pkl", "rb") as f:
                    st.download_button("📦 Download Trained Model", f, file_name="model.pkl")
        except Exception as e:
            st.error(f"❌ Error: {e}")

#PREDICTION MODULE

st.header("🔮 Predict Using Trained Model")

uploaded_model = st.file_uploader("📦 Upload Your Trained Model (.pkl)", type=["pkl"], key="model_upload")
new_data_file = st.file_uploader("📄 Upload New CSV for Prediction", type=["csv"], key="data_upload")

if uploaded_model and new_data_file:
    try:
        loaded_obj = pickle.load(uploaded_model)

        if isinstance(loaded_obj, tuple):
            model, expected_cols = loaded_obj
        else:
            model = loaded_obj
            expected_cols = None

        new_data = pd.read_csv(new_data_file)
        new_data = pd.get_dummies(new_data)

        if expected_cols:
            for col in expected_cols:
                if col not in new_data.columns:
                    new_data[col] = 0
            new_data = new_data[[col for col in new_data.columns if col in expected_cols]]
            new_data = new_data.reindex(columns=expected_cols)

        predictions = model.predict(new_data)
        new_data["Prediction"] = predictions

        st.success("✅ Predictions completed.")
        st.write("📊 Predicted Data", new_data.head())

        csv = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
