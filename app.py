import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('salary_data.csv')

le_role = LabelEncoder()
le_demand = LabelEncoder()

df['job_role_encoded'] = le_role.fit_transform(df['job_role'])
df['market_demand_encoded'] = le_demand.fit_transform(df['market_demand'])

X = df[['experience', 'job_role_encoded', 'market_demand_encoded']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Streamlit UI
st.title("💼 Salary Negotiation Predictor")

experience = st.slider("Years of Experience", 0, 20, 3)
job_role = st.selectbox("Job Role", df['job_role'].unique())
market_demand = st.selectbox("Market Demand", df['market_demand'].unique())

# Encode input
input_data = pd.DataFrame({
    'experience': [experience],
    'job_role_encoded': [le_role.transform([job_role])[0]],
    'market_demand_encoded': [le_demand.transform([market_demand])[0]]
})

# Predict
predicted_salary = model.predict(input_data)[0]

st.success(f"💰 Estimated Negotiable Salary: ₹{int(predicted_salary):,}/year")
