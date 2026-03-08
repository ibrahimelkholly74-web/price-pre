import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel("laptops_training_dataset.xlsx")

le_brand = LabelEncoder()
le_condition = LabelEncoder()

df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Condition'] = le_condition.fit_transform(df['Condition'])

X = df.drop('Price', axis=1)
y = df['Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("💻 Laptop Price Predictor")

brand = st.selectbox("Brand", le_brand.classes_)
ram = st.selectbox("RAM", [4,8,16,32])
storage = st.selectbox("Storage", [128,256,512,1024])
cpu = st.selectbox("CPU Generation", [4,5,6,7,8,9,10,11])
year = st.selectbox("Year", list(range(2018,2027)))
condition = st.selectbox("Condition", le_condition.classes_)

if st.button("Predict Price"):
    brand_code = le_brand.transform([brand])[0]
    condition_code = le_condition.transform([condition])[0]

    new_data = pd.DataFrame({
        'Brand':[brand_code],
        'RAM':[ram],
        'Storage':[storage],
        'CPU_Gen':[cpu],
        'Year':[year],
        'Condition':[condition_code]
    })

    price = model.predict(new_data)[0]
    st.success(f"Predicted Price: {price:.0f} EGP")
