import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime
from io import BytesIO

# Set Streamlit Page Configuration
st.set_page_config(page_title="Inventory Management", layout="wide", initial_sidebar_state="expanded")

# Session State Initialization
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'staff'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dark Mode Toggle
if st.sidebar.checkbox("🌙 Dark Mode"):
    st.session_state.dark_mode = True
    st.markdown("<style>body { background-color: #111; color: #EEE; }</style>", unsafe_allow_html=True)

# User Login (Basic Simulation)
st.sidebar.title("🔐 Login")
user = st.sidebar.text_input("Username")
role = st.sidebar.selectbox("Role", ["admin", "staff"])
if st.sidebar.button("Login"):
    st.session_state.user_role = role
    st.success(f"Logged in as {user} ({role})")

# Sidebar Navigation
st.sidebar.title("MENU")
menu = st.sidebar.radio("Go to", ["Dashboard", "Inventory", "Cashier", "Reports", "Suppliers"])

# Function to load data
@st.cache_data
def load_data(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == "csv":
        df = pd.read_csv(file)
    elif file_extension in ["xlsm", "xlsx"]:
        df = pd.read_excel(file, engine='openpyxl')
    else:
        st.error("Unsupported file format. Upload CSV or Excel files only.")
        return None
    return df

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload Inventory Data (CSV/XLSM)", type=["csv", "xlsm"])
df = load_data(uploaded_file) if uploaded_file else None

# Admin Panel
if st.session_state.user_role == "admin":
    with st.expander("⚙️ Admin Tools"):
        st.subheader("📁 Upload New Inventory")
        new_file = st.file_uploader("Upload New Inventory File", type=["csv", "xlsm"], key="admin_upload")
        if new_file:
            df = load_data(new_file)
            st.success("New inventory loaded successfully!")

        if df is not None:
            st.subheader("⬇️ Export Current Inventory")
            st.download_button("Download Inventory CSV", df.to_csv(index=False), file_name="exported_inventory.csv", mime="text/csv")

            st.subheader("🏪 Manage Supplier Info")
            supplier_name = st.text_input("Supplier Name")
            supplier_id = st.text_input("Supplier ID")
            location = st.text_input("Warehouse Location")
            if st.button("Add Supplier"):
                new_supplier = pd.DataFrame([{
                    'Supplier_Name': supplier_name,
                    'Supplier_ID': supplier_id,
                    'Warehouse_Location': location
                }])
                df = pd.concat([df, new_supplier], ignore_index=True)
                st.success(f"Supplier {supplier_name} added successfully!")

# Dashboard
if menu == "Dashboard":
    st.title("📊 Inventory Dashboard")
    if df is not None:
        st.write("### Current Stock Overview")
        st.dataframe(df[['Product_Name', 'Stock_Quantity', 'Reorder_Level', 'Unit_Price']])

        low_stock = df[df['Stock_Quantity'] < df['Reorder_Level']]
        if not low_stock.empty:
            st.warning(f"⚠️ {len(low_stock)} products are below the reorder level!")

        if 'Inventory_Turnover_Rate' in df.columns:
            st.write("### Inventory Turnover Analysis")
            fig = px.bar(df, x='Product_Name', y='Inventory_Turnover_Rate', title="Turnover Rate by Product")
            st.plotly_chart(fig)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Current Inventory", csv, "inventory.csv", "text/csv")
    else:
        st.info("Please upload inventory data.")

# Inventory Management
elif menu == "Inventory":
    st.title("📦 Inventory Management")
    if df is not None:
        st.write("### All Inventory Items")
        df_edit = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        search = st.text_input("🔍 Search Product")
        if search:
            results = df[df['Product_Name'].str.contains(search, case=False, na=False)]
            st.dataframe(results)
        if st.session_state.user_role == "admin":
            st.success("✔️ Editable table enabled for admins")
            st.download_button("💾 Download Edited Inventory", df_edit.to_csv(index=False), "updated_inventory.csv", "text/csv")
    else:
        st.info("Upload data first.")

# Cashier System
elif menu == "Cashier":
    st.title("🛒 Cashier System")
    if df is not None:
        product = st.selectbox("Select a product", df['Product_Name'])
        quantity = st.number_input("Enter Quantity", min_value=1, value=1)
        product_row = df[df['Product_Name'] == product]
        if not product_row.empty and 'Unit_Price' in product_row.columns:
            price = product_row['Unit_Price'].values[0]
            total = quantity * price
            st.metric("Total", f"${total:.2f}")

            if st.button("Add to Cart"):
                st.session_state.cart.append({
                    "Product": product,
                    "Quantity": quantity,
                    "Unit Price": price,
                    "Total": total
                })
                st.success(f"{quantity} x {product} added to cart!")

            if st.session_state.cart:
                st.write("### 🧺 Cart Summary")
                cart_df = pd.DataFrame(st.session_state.cart)
                st.dataframe(cart_df)
                grand_total = cart_df['Total'].sum()
                st.subheader(f"🧾 Grand Total: ${grand_total:.2f}")
        else:
            st.error("Selected product doesn't have a valid price.")
    else:
        st.info("Upload inventory data first.")

# Reports & Analysis
elif menu == "Reports":
    st.title("📈 Sales Analysis & Demand Forecasting")
    if df is not None:
        if 'Last_Order_Date' in df.columns:
            df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'], errors='coerce')
            fig = px.line(df, x='Last_Order_Date', y='Sales_Volume', title="Sales Trends")
            st.plotly_chart(fig)

        if 'Date_Received' in df.columns and 'Sales_Volume' in df.columns:
            df['Date_Received'] = pd.to_datetime(df['Date_Received'], errors='coerce')
            df['Days_Since_Received'] = (datetime.today() - df['Date_Received']).dt.days
            X = df[['Days_Since_Received']].dropna()
            y = df['Sales_Volume'].dropna()
            if not X.empty and not y.empty:
                model = LinearRegression()
                model.fit(X, y)
                future_days = np.array([[30], [60], [90]])
                predictions = model.predict(future_days)
                st.write("### 🔮 Demand Forecasting")
                for i, day in enumerate([30, 60, 90]):
                    st.write(f"Predicted Sales in {day} days: **{predictions[i]:.2f}** units")
    else:
        st.info("Upload sales data first.")

# Supplier Management
elif menu == "Suppliers":
    st.title("🏪 Supplier Management")
    if df is not None:
        st.dataframe(df[['Supplier_Name', 'Supplier_ID', 'Warehouse_Location']])
    else:
        st.info("Upload inventory data first.")
