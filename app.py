# Smart Inventory Management System with Advanced Features
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime
import hashlib

# ------------------------
# Helper functions
# ------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Dummy users database (for demo)
users = {
    "admin": hash_password("admin123"),
    "staff": hash_password("staffpass")
}

# Auth
if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.username = ""

if not st.session_state.auth:
    with st.form("Login"):
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

        if login:
            if username in users and hash_password(password) == users[username]:
                st.session_state.auth = True
                st.session_state.username = username
                st.success("Login successful")
            else:
                st.error("Invalid credentials")
    st.stop()

# ------------------------
# App config and theme
# ------------------------
st.set_page_config(page_title="Smart Inventory", layout="wide")

# Dark/Light mode
st.sidebar.markdown("## Theme")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown('<style>body { background-color: #1e1e1e; color: white; }</style>', unsafe_allow_html=True)

# Upload
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Inventory File", type=["csv", "xlsx"])
menu = st.sidebar.radio("Navigate", ["Dashboard", "Inventory", "Cashier", "Reports", "Suppliers"])

@st.cache_data

def load_data(file):
    ext = file.name.split('.')[-1]
    if ext == "csv":
        return pd.read_csv(file)
    return pd.read_excel(file, engine='openpyxl')

df = load_data(uploaded_file) if uploaded_file else None

# ------------------------
# Dashboard
# ------------------------
if menu == "Dashboard":
    st.title("📦 Inventory Dashboard")
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Products", df['Product_Name'].nunique())
        col2.metric("Total Stock", int(df['Stock_Quantity'].sum()))
        low_stock = df[df['Stock_Quantity'] < df['Reorder_Level']]
        col3.metric("Low Stock Items", low_stock.shape[0])

        st.markdown("### 📈 Stock Levels")
        fig = px.bar(df, x='Product_Name', y='Stock_Quantity', color='Stock_Quantity', height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📥 Download Inventory")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "inventory.csv", "text/csv")
    else:
        st.info("Upload a file to begin.")

# ------------------------
# Inventory
# ------------------------
elif menu == "Inventory":
    st.title("🗃️ Inventory Viewer")
    if df is not None:
        edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        st.download_button("Download Updated Data", edited_df.to_csv(index=False).encode(), "updated_inventory.csv")
    else:
        st.info("Upload a file to begin.")

# ------------------------
# Cashier
# ------------------------
elif menu == "Cashier":
    st.title("🛒 Cashier")
    if df is not None:
        if 'cart' not in st.session_state:
            st.session_state.cart = []

        product = st.selectbox("Select Product", df['Product_Name'].unique())
        quantity = st.number_input("Quantity", min_value=1)

        price = df[df['Product_Name'] == product]['Unit_Price'].values[0]
        total = quantity * price
        st.markdown(f"### 💵 Total: ${total:.2f}")

        if st.button("Add to Cart"):
            st.session_state.cart.append({"Product": product, "Qty": quantity, "Total": total})
            st.success("Added to cart")

        if st.session_state.cart:
            st.markdown("### 🧾 Cart")
            st.dataframe(pd.DataFrame(st.session_state.cart))
    else:
        st.info("Upload inventory data first.")

# ------------------------
# Reports & Forecast
# ------------------------
elif menu == "Reports":
    st.title("📊 Reports & Forecast")
    if df is not None and 'Sales_Volume' in df.columns and 'Date_Received' in df.columns:
        df['Date_Received'] = pd.to_datetime(df['Date_Received'], errors='coerce')
        df['Days_Since_Received'] = (datetime.today() - df['Date_Received']).dt.days

        df.dropna(subset=['Sales_Volume', 'Days_Since_Received'], inplace=True)
        X = df[['Days_Since_Received']]
        y = df['Sales_Volume']

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array([[30], [60], [90]])
        preds = model.predict(future_days)

        st.subheader("🔮 Forecasted Sales")
        forecast_df = pd.DataFrame({
            'Days From Now': [30, 60, 90],
            'Predicted Volume': preds.round(2)
        })
        st.table(forecast_df)
    else:
        st.warning("Forecasting needs 'Sales_Volume' and 'Date_Received' columns.")

# ------------------------
# Suppliers
# ------------------------
elif menu == "Suppliers":
    st.title("🏪 Suppliers")
    if df is not None:
        st.dataframe(df[['Supplier_Name', 'Supplier_ID', 'Warehouse_Location']], use_container_width=True)
        with st.expander("Add Supplier"):
            name = st.text_input("Name")
            sid = st.text_input("ID")
            location = st.text_input("Location")
            if st.button("Add"):
                df.loc[len(df)] = [None]*len(df.columns)  # Add empty row
                df.at[df.index[-1], 'Supplier_Name'] = name
                df.at[df.index[-1], 'Supplier_ID'] = sid
                df.at[df.index[-1], 'Warehouse_Location'] = location
                st.success("Supplier added")
    else:
        st.info("Upload data first.")
