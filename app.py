import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Set page config
st.set_page_config(page_title="Inventory Management", layout="wide")

# Sidebar Navigation
st.sidebar.title("MENU")
menu = st.sidebar.radio("Go to", ["Dashboard", "Inventory", "Cashier", "Reports", "Suppliers"])

# Function to load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file, engine='openpyxl')
    return df

# Upload File
uploaded_file = st.sidebar.file_uploader("Upload Inventory Data (CSV/XLSM)", type=["csv", "xlsm"])
if uploaded_file:
    df = load_data(uploaded_file)

# Dashboard
if menu == "Dashboard":
    st.title("📊 Inventory Dashboard")

    if uploaded_file:
        st.write("### Current Stock Overview")
        st.dataframe(df[['Product_Name', 'Stock_Quantity', 'Reorder_Level', 'Unit_Price']])

        # Stock Status
        low_stock = df[df['Stock_Quantity'] < df['Reorder_Level']]
        st.warning(f"⚠️ {len(low_stock)} products are below the reorder level!")

        # Inventory Turnover Rate Analysis
        st.write("### Inventory Turnover Analysis")
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Product_Name', y='Inventory_Turnover_Rate', data=df)
        plt.xticks(rotation=90)
        st.pyplot(plt)
    else:
        st.info("Please upload inventory data.")

# Inventory Management
elif menu == "Inventory":
    st.title("📦 Inventory Management")

    if uploaded_file:
        st.dataframe(df)
        search = st.text_input("Search Product", "")
        if search:
            results = df[df['Product_Name'].str.contains(search, case=False, na=False)]
            st.write("### Search Results")
            st.dataframe(results)
    else:
        st.info("Upload data first.")

# Cashier System
elif menu == "Cashier":
    st.title("🛒 Cashier System")

    if uploaded_file:
        st.write("### Select Products")
        product = st.selectbox("Select a product", df['Product_Name'])
        quantity = st.number_input("Enter Quantity", min_value=1, value=1)
        price = df[df['Product_Name'] == product]['Unit_Price'].values[0]
        total = quantity * price
        st.write(f"Total: **${total:.2f}**")
        
        if st.button("Add to Cart"):
            st.success(f"{quantity} x {product} added to cart!")
    else:
        st.info("Upload inventory data first.")

# Reports & Analysis
elif menu == "Reports":
    st.title("📈 Sales Analysis & Demand Forecasting")

    if uploaded_file:
        st.write("### Sales Trends")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='Last_Order_Date', y='Sales_Volume', data=df)
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Demand Forecasting using Linear Regression
        st.write("### Demand Forecasting")
        df['Date_Received'] = pd.to_datetime(df['Date_Received'])
        df['Days_Since_Received'] = (datetime.today() - df['Date_Received']).dt.days
        X = df[['Days_Since_Received']]
        y = df['Sales_Volume']
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.array([[30], [60], [90]])
        predictions = model.predict(future_days)
        st.write(f"Predicted Sales in 30 days: **{predictions[0]:.2f}** units")
        st.write(f"Predicted Sales in 60 days: **{predictions[1]:.2f}** units")
        st.write(f"Predicted Sales in 90 days: **{predictions[2]:.2f}** units")
    else:
        st.info("Upload sales data first.")

# Supplier Management
elif menu == "Suppliers":
    st.title("🏪 Supplier Management")
    
    if uploaded_file:
        st.dataframe(df[['Supplier_Name', 'Supplier_ID', 'Warehouse_Location']])
        supplier_name = st.text_input("Supplier Name")
        supplier_id = st.text_input("Supplier ID")
        location = st.text_input("Warehouse Location")

        if st.button("Add Supplier"):
            new_supplier = {'Supplier_Name': supplier_name, 'Supplier_ID': supplier_id, 'Warehouse_Location': location}
            df = df.append(new_supplier, ignore_index=True)
            st.success(f"Supplier {supplier_name} added successfully!")
    else:
        st.info("Upload inventory data first.")
