import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---- Database Connection ----
conn = sqlite3.connect('inventory.db', check_same_thread=False)
cursor = conn.cursor()

# ---- Create Table if Not Exists ----
cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    stock_level INTEGER NOT NULL,
    sales_history TEXT NOT NULL
)
""")
conn.commit()

# ---- Load or Train ML Model ----
def train_model():
    cursor.execute("SELECT * FROM inventory")
    data = cursor.fetchall()

    if not data:
        return None

    df = pd.DataFrame(data, columns=['id', 'product_name', 'stock_level', 'sales_history'])
    df['sales_history'] = df['sales_history'].apply(lambda x: list(map(int, x.split(','))))

    X, y = [], []
    for _, row in df.iterrows():
        sales = row['sales_history']
        if len(sales) > 3:  # Ensure enough data points
            X.append(sales[:-1])  # Features: past sales
            y.append(sales[-1])  # Target: next sales

    if not X or not y:
        return None

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "demand_forecast_model.pkl")
    return model

def load_model():
    try:
        return joblib.load("demand_forecast_model.pkl")
    except:
        return train_model()

model = load_model()

# ---- Streamlit UI ----
st.set_page_config(page_title="Inventory Management", layout="wide")

st.title("📦 AI-Powered Inventory Management System")
st.markdown("Manage stock levels, predict demand, and analyze inventory trends!")

menu = ["Dashboard", "Manage Inventory", "Forecast Demand", "Analysis"]
choice = st.sidebar.radio("Navigation", menu)

# ---- Dashboard ----
if choice == "Dashboard":
    st.subheader("📊 Inventory Overview")

    cursor.execute("SELECT * FROM inventory")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['ID', 'Product Name', 'Stock Level', 'Sales History'])

    if not df.empty:
        df['Sales History'] = df['Sales History'].apply(lambda x: list(map(int, x.split(','))))
        st.dataframe(df)

        st.subheader("📉 Stock Level Trends")
        for _, row in df.iterrows():
            plt.plot(row['Sales History'], label=row['Product Name'])

        plt.xlabel("Time (days)")
        plt.ylabel("Stock Level")
        plt.legend()
        st.pyplot(plt)

# ---- Manage Inventory ----
elif choice == "Manage Inventory":
    st.subheader("🛒 Add / Edit Inventory")

    with st.form("inventory_form"):
        product_name = st.text_input("Product Name")
        stock_level = st.number_input("Stock Level", min_value=0, step=1)
        sales_history = st.text_area("Sales History (comma-separated)", placeholder="e.g., 10,15,12,18")

        submit = st.form_submit_button("Add Product")

        if submit:
            cursor.execute("INSERT INTO inventory (product_name, stock_level, sales_history) VALUES (?, ?, ?)",
                           (product_name, stock_level, sales_history))
            conn.commit()
            st.success(f"{product_name} added successfully!")

    st.subheader("📜 Current Inventory")
    cursor.execute("SELECT * FROM inventory")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['ID', 'Product Name', 'Stock Level', 'Sales History'])

    if not df.empty:
        st.dataframe(df)

        delete_id = st.number_input("Enter Product ID to Delete:", min_value=0, step=1)
        delete_button = st.button("Delete Product")

        if delete_button:
            cursor.execute("DELETE FROM inventory WHERE id=?", (delete_id,))
            conn.commit()
            st.warning("Product deleted!")

# ---- Forecast Demand ----
elif choice == "Forecast Demand":
    st.subheader("📈 Predict Future Demand")

    cursor.execute("SELECT * FROM inventory")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['ID', 'Product Name', 'Stock Level', 'Sales History'])

    if not df.empty:
        product_id = st.selectbox("Select Product ID", df['ID'])
        selected_product = df[df['ID'] == product_id].iloc[0]

        sales_data = list(map(int, selected_product['Sales History'].split(',')))
        st.write(f"**Sales Data:** {sales_data}")

        if len(sales_data) > 3 and model:
            input_data = np.array(sales_data[-3:]).reshape(1, -1)
            prediction = model.predict(input_data)
            st.success(f"📊 Predicted demand for next cycle: {int(prediction[0])}")
        else:
            st.error("Not enough data for forecasting. Add more sales history!")

# ---- Analysis ----
elif choice == "Analysis":
    st.subheader("📊 Sales & Inventory Analysis")

    cursor.execute("SELECT * FROM inventory")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['ID', 'Product Name', 'Stock Level', 'Sales History'])

    if not df.empty:
        df['Sales History'] = df['Sales History'].apply(lambda x: list(map(int, x.split(','))))
        avg_stock = df['Stock Level'].mean()
        avg_sales = df['Sales History'].apply(np.mean).mean()

        col1, col2 = st.columns(2)
        col1.metric("📦 Avg Stock Level", f"{avg_stock:.2f}")
        col2.metric("📈 Avg Sales per Product", f"{avg_sales:.2f}")

        st.subheader("🔹 Stock vs Sales Trends")
        for _, row in df.iterrows():
            plt.plot(row['Sales History'], label=row['Product Name'])

        plt.xlabel("Time (days)")
        plt.ylabel("Stock Level / Sales")
        plt.legend()
        st.pyplot(plt)

# ---- Run the App ----
if __name__ == "__main__":
    st.write("")
