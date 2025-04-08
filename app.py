import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Page Config
st.set_page_config(page_title="Smart Inventory", layout="wide")

# Enhanced CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        background-color: #0066cc;
        color: white;
    }
    .stMetric {
        background: #e8f0fe;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
    }
    .css-1d391kg { padding: 3rem !important; }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'staff'

# Sidebar Login and Menu
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1041/1041916.png", width=80)
    st.title("Smart Inventory")
    user = st.text_input("👤 Username", value="admin")
    role = st.selectbox("🔐 Role", ["admin", "staff"])
    if st.button("🔓 Login"):
        st.session_state.user_role = role
        st.success(f"Logged in as {user} ({role})")

    st.markdown("---")
    menu = st.radio("📂 Navigation", ["Dashboard", "Inventory", "Cashier", "Reports", "Suppliers", "Customer Feedback"])

# Load data
@st.cache_data

def load_data(file):
    ext = file.name.split('.')[-1]
    if ext == "csv":
        return pd.read_csv(file)
    elif ext in ["xlsm", "xlsx"]:
        return pd.read_excel(file, engine='openpyxl')
    st.error("Unsupported file format.")
    return None

uploaded_file = st.sidebar.file_uploader("📁 Upload Inventory Data", type=["csv", "xlsm"])
df = load_data(uploaded_file) if uploaded_file else None

# Admin Controls
if st.session_state.user_role == "admin":
    with st.expander("🛠️ Admin Tools", expanded=False):
        new_file = st.file_uploader("📤 Upload New Inventory", type=["csv", "xlsm"], key="admin_upload")
        if new_file:
            df = load_data(new_file)
            st.success("✅ New inventory loaded!")
        if df is not None:
            st.download_button("⬇️ Download Inventory", df.to_csv(index=False), "inventory.csv")

# Dashboard
if menu == "Dashboard":
    st.title("📊 Dashboard Overview")
    if df is not None:
        st.subheader("📋 Inventory Summary")
        st.dataframe(df[['Product_Name', 'Stock_Quantity', 'Reorder_Level', 'Unit_Price']], use_container_width=True)

        low_stock = df[df['Stock_Quantity'] < df['Reorder_Level']]
        if not low_stock.empty:
            st.warning(f"⚠️ {len(low_stock)} products are below reorder level")

        if 'Inventory_Turnover_Rate' in df.columns:
            st.plotly_chart(px.bar(df, x='Product_Name', y='Inventory_Turnover_Rate', title="📈 Inventory Turnover Rate"))
    else:
        st.info("Please upload inventory data to proceed.")

# Inventory Management
elif menu == "Inventory":
    st.title("📦 Inventory Management")
    if df is not None:
        st.data_editor(df, use_container_width=True, num_rows="dynamic")
        search = st.text_input("🔍 Search Product")
        if search:
            st.dataframe(df[df['Product_Name'].str.contains(search, case=False)])
        if st.session_state.user_role == "admin":
            st.download_button("💾 Save Inventory", df.to_csv(index=False), "updated_inventory.csv")
    else:
        st.info("Please upload data to manage inventory.")

# Cashier
elif menu == "Cashier":
    st.title("🛒 Point of Sale (Cashier)")
    if df is not None:
        product = st.selectbox("🛍️ Select Product", df['Product_Name'])
        quantity = st.number_input("🔢 Quantity", min_value=1, value=1)
        row = df[df['Product_Name'] == product]
        if not row.empty:
            try:
                price = float(row['Unit_Price'].values[0])
                total = quantity * price
                st.metric("🧾 Total", f"${total:.2f}")
                if st.button("➕ Add to Cart"):
                    st.session_state.cart.append({"Product": product, "Quantity": quantity, "Unit Price": price, "Total": total})
                    st.success(f"Added {quantity} x {product} to cart")
                if st.session_state.cart:
                    cart_df = pd.DataFrame(st.session_state.cart)
                    st.dataframe(cart_df)
                    st.subheader(f"💰 Grand Total: ${cart_df['Total'].sum():.2f}")
            except Exception as e:
                st.error(f"❌ Error calculating total: {e}")
    else:
        st.info("Upload inventory data first.")

# Reports
elif menu == "Reports":
    st.title("📈 Sales & Forecasting Reports")
    if df is not None:
        if 'Last_Order_Date' in df.columns:
            df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'], errors='coerce')
            st.plotly_chart(px.line(df, x='Last_Order_Date', y='Sales_Volume', title="📊 Sales Trends"))

        if 'Date_Received' in df.columns and 'Sales_Volume' in df.columns:
            df['Date_Received'] = pd.to_datetime(df['Date_Received'], errors='coerce')
            df['Days_Since_Received'] = (datetime.today() - df['Date_Received']).dt.days
            X = df[['Days_Since_Received']].dropna()
            y = df['Sales_Volume'].dropna()
            if not X.empty and not y.empty:
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(np.array([[30], [60], [90]]))
                st.write("### 🔮 30-90 Day Demand Forecast")
                for d, p in zip([30, 60, 90], predictions):
                    st.write(f"In {d} days: **{p:.2f}** units")
    else:
        st.info("Upload inventory data to see reports.")

# Suppliers
elif menu == "Suppliers":
    st.title("🏭 Supplier Overview")
    if df is not None:
        st.dataframe(df[['Supplier_Name', 'Supplier_ID', 'Warehouse_Location']])
    else:
        st.info("Please upload data first.")

# Customer Feedback
elif menu == "Customer Feedback":
    st.title("💬 Customer Feedback Portal")
    with st.form("feedback_form", clear_on_submit=True):
        st.subheader("📝 Leave Your Feedback")
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Feedback")
        rating = st.slider("Rating", 1, 5, 3)
        submit = st.form_submit_button("Submit")
        if submit:
            fb = {"Name": name, "Email": email, "Feedback": message, "Rating": rating, "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            if "feedback_data" not in st.session_state:
                st.session_state.feedback_data = []
            st.session_state.feedback_data.append(fb)
            st.success("🎉 Thanks for your feedback!")

    if st.session_state.user_role == "admin":
        st.subheader("📋 All Feedback")
        if "feedback_data" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state.feedback_data))
        else:
            st.info("No feedback submitted yet.")
