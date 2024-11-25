import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load datasets (Caching the function to improve app performance)
@st.cache_data
def load_data():
    patient_demo = pd.read_excel("data/Patient_Demo.xlsx")
    patient_hospital_visit = pd.read_excel("data/Patient_Hospital_Visit.xlsx")
    return patient_demo, patient_hospital_visit

# Load and merge datasets
patient_demo, patient_hospital_visit = load_data()
data = pd.merge(patient_demo, patient_hospital_visit, on="Patient_ID", how="inner")

# Streamlit app structure
st.title("Resource Allocation Prediction App")
st.write("This application predicts and allocates resources based on patient data and user-defined criteria.")

# Sidebar: Collect user inputs
st.sidebar.header("User Input Parameters")

patient_id = st.sidebar.selectbox("Select Patient ID", data["Patient_ID"].unique())
hospital_capacity = st.sidebar.number_input("Hospital Capacity (beds)", min_value=10, max_value=1000, value=100)
total_days = st.sidebar.number_input("Total Days to Allocate Resources", min_value=1, max_value=30, value=7)
length_of_stay_input = st.sidebar.number_input("Length of Stay (days)", min_value=1, max_value=30, value=7)
resource_usage_per_day_input = st.sidebar.number_input("Resource Usage Per Day", min_value=1, max_value=100, value=5)

# Filter data for the selected patient
patient_data = data[data["Patient_ID"] == patient_id]

# Display selected patient data
st.write("### Selected Patient Data")
st.write(patient_data)

# Improved Prediction Model using RandomForest
def train_model(data, model_type='random_forest'):
    # Features for prediction
    features = ['Age', 'Visit_Frequency', 'Length_of_Stay']
    target = 'Resource_Usage_Per_Day'

    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Model Evaluation (MAE): {mae:.2f}")
    
    return model

# Train Random Forest Model
model = train_model(data, model_type='random_forest')

# Make prediction based on selected patient
if st.button("Predict"):
    # Extract patient features
    patient_features = patient_data[['Age', 'Visit_Frequency', 'Length_of_Stay']].values
    predicted_resource_usage = model.predict(patient_features)[0]

    # Display predicted resource usage
    st.write(f"### Predicted Resource Usage Per Day: {predicted_resource_usage:.2f}")

    # Resource Allocation Logic
    total_resources = predicted_resource_usage * length_of_stay_input
    daily_allocation = total_resources / total_days

    # Generate allocation schedule
    allocation_schedule = pd.DataFrame({
        "Day": range(1, total_days + 1),
        "Allocated Resources": [daily_allocation] * total_days
    })

    # Display allocation schedule
    st.write("### Resource Allocation Schedule")
    st.write(allocation_schedule)

    # Visualization: Allocation over time
    st.write("### Daily Resource Allocation Chart")
    fig, ax = plt.subplots()
    sns.lineplot(x="Day", y="Allocated Resources", data=allocation_schedule, ax=ax)
    ax.set_title("Daily Resource Allocation")
    ax.set_xlabel("Day")
    ax.set_ylabel("Allocated Resources")
    st.pyplot(fig)

    # Visualization: Capacity vs. Predicted Requirement
    st.write("### Hospital Capacity vs. Predicted Resource Requirement")
    fig, ax = plt.subplots()
    sns.barplot(x=["Predicted Requirement", "Hospital Capacity"],
                y=[total_resources, hospital_capacity], ax=ax)
    ax.set_title("Hospital Capacity vs. Requirement")
    ax.set_ylabel("Resources")
    st.pyplot(fig)

    # Visualization: Resource Overage
    st.write("### Resource Overage Chart")
    overage = total_resources - hospital_capacity
    if overage > 0:
        st.write(f"Warning: Hospital is over-allocated by {overage:.2f} resources.")
        fig, ax = plt.subplots()
        sns.barplot(x=["Required Resources", "Hospital Capacity", "Overage"],
                    y=[total_resources, hospital_capacity, overage], ax=ax)
        ax.set_title("Resource Overage Comparison")
        ax.set_ylabel("Resources")
        st.pyplot(fig)
    else:
        st.write("Hospital capacity is sufficient to meet the resource requirements.")
