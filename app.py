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
            st.subheader("🔄 Inventory Turnover Rate by Product")

            # Sort data for better readability
            turnover_sorted = df.sort_values(by='Inventory_Turnover_Rate', ascending=False)

            fig_turnover = px.bar(
                turnover_sorted,
                x='Product_Name',
                y='Inventory_Turnover_Rate',
                color='Inventory_Turnover_Rate',
                title="📈 Inventory Turnover Rate",
                labels={
                    'Product_Name': 'Product',
                    'Inventory_Turnover_Rate': 'Turnover Rate'
                },
                text='Inventory_Turnover_Rate'
            )

            # Make layout more readable
            fig_turnover.update_layout(
                xaxis_tickangle=-45,
                xaxis_tickfont=dict(size=10),
                yaxis_title="Turnover Rate",
                xaxis_title="Product",
                plot_bgcolor="#f9f9f9",
                bargap=0.4,
                height=500
            )
            fig_turnover.update_traces(texttemplate='%{text:.2f}', textposition='outside')

            st.plotly_chart(fig_turnover, use_container_width=True)

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
    # Ensure df is loaded

# Check if the dataframe is loaded and contains data
    if df is not None and not df.empty:
        # Product Selection
        try:
            product_names = df['Product_Name'].dropna().unique()
            
            if len(product_names) == 0:
                st.warning("⚠️ No products available in the uploaded data.")
            else:
                # Create a layout with columns for a clean display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Product selection dropdown
                    product = st.selectbox("🛍️ Select Product", product_names)
                    
                with col2:
                    # Quantity input
                    quantity = st.number_input("🔢 Quantity", min_value=1, value=1, step=1)
                
                # Get the product row from the dataframe
                row = df[df['Product_Name'] == product]
                
                if not row.empty:
                    try:
                        # Clean and convert price to float (removing '$' symbol)
                        price_str = row['Unit_Price'].values[0]
                        price_str_cleaned = price_str.replace('$', '').strip()  # Remove '$' and spaces
                        price = float(price_str_cleaned)  # Convert cleaned string to float
                        total = quantity * price

                        # Display the total dynamically
                        st.metric("🧾 Total", f"₹{total:.2f}", delta=None)

                        # Add to cart button with success message
                        if st.button("➕ Add to Cart", key=f"{product}_add"):
                            st.session_state.cart.append({
                                "Product": product,
                                "Quantity": quantity,
                                "Unit Price": price,
                                "Total": total
                            })
                            st.success(f"✅ Added {quantity} x {product} to cart")

                        # Display Cart Section
                        if st.session_state.cart:
                            st.subheader("🛒 Cart Details")
                            cart_df = pd.DataFrame(st.session_state.cart)
                            st.dataframe(cart_df, use_container_width=True)

                            grand_total = cart_df["Total"].sum()
                            st.subheader(f"💰 Grand Total: ₹{grand_total:.2f}")

                    except Exception as e:
                        st.error(f"❌ Error calculating price: {e}")
                else:
                    st.error("❌ Selected product not found in data.")
        except KeyError:
            st.error("❌ 'Product_Name' or 'Unit_Price' column missing in uploaded file.")
    else:
        st.info("📂 Please upload inventory data first.")

# Reports
# Reports
elif menu == "Reports":
    st.title("📈 Sales & Forecasting Reports")

    if df is not None and not df.empty:

        # Sales Trend by Date
        if 'Last_Order_Date' in df.columns and 'Sales_Volume' in df.columns:
            df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'], errors='coerce')
            sales_trend = df.groupby('Last_Order_Date')['Sales_Volume'].sum().reset_index()
            st.subheader("📊 Sales Volume Over Time")
            fig_trend = px.line(sales_trend, x='Last_Order_Date', y='Sales_Volume', markers=True,
                                title="Sales Trend by Date",
                                labels={'Sales_Volume': 'Units Sold'})
            st.plotly_chart(fig_trend, use_container_width=True)

        # Top Selling Products
        if 'Product_Name' in df.columns and 'Sales_Volume' in df.columns:
            top_products = df.groupby('Product_Name')['Sales_Volume'].sum().sort_values(ascending=False).head(10)
            st.subheader("🏆 Top 10 Selling Products")
            fig_top = px.bar(top_products, x=top_products.values, y=top_products.index,
                             orientation='h', color=top_products.values,
                             labels={'x': 'Units Sold', 'y': 'Product'},
                             title="Top Selling Products")
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)

        # Inventory by Category
        if 'Category' in df.columns and 'Stock_Quantity' in df.columns:
            st.subheader("📦 Inventory by Category")
            cat_inventory = df.groupby('Category')['Stock_Quantity'].sum().reset_index()
            fig_category = px.pie(cat_inventory, names='Category', values='Stock_Quantity',
                                  title="Inventory Distribution by Category")
            st.plotly_chart(fig_category, use_container_width=True)

        # Low Stock Alert
        if 'Product_Name' in df.columns and 'Stock_Quantity' in df.columns and 'Reorder_Level' in df.columns:
            st.subheader("🚨 Low Stock Alerts")
            low_stock = df[df['Stock_Quantity'] < df['Reorder_Level']]
            st.write(f"🧯 {len(low_stock)} products below reorder level:")
            st.dataframe(low_stock[['Product_Name', 'Stock_Quantity', 'Reorder_Level']], use_container_width=True)

        # Reorder Quantity Heatmap
        if all(col in df.columns for col in ['Warehouse_Location', 'Reorder_Quantity', 'Category']):
            st.subheader("🔥 Reorder Quantity Heatmap by Warehouse")
            pivot = df.pivot_table(values='Reorder_Quantity', index='Warehouse_Location',
                                   columns='Category', aggfunc='sum').fillna(0)
            st.dataframe(pivot.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
        else:
            st.info("📛 Missing columns for Reorder Quantity Heatmap (Need: Warehouse_Location, Reorder_Quantity, Category)")

        # Supplier Distribution
        if 'Supplier_Name' in df.columns and 'Stock_Quantity' in df.columns:
            st.subheader("🏢 Supplier-wise Product Distribution")
            supplier_dist = df.groupby('Supplier_Name')['Stock_Quantity'].sum().reset_index()
            fig_supplier = px.bar(supplier_dist, x='Supplier_Name', y='Stock_Quantity',
                                  title="Stock Supplied by Each Supplier",
                                  labels={'Stock_Quantity': 'Units in Stock'})
            st.plotly_chart(fig_supplier, use_container_width=True)

        
        # Forecasting for Specific Product
        if all(col in df.columns for col in ['Product_Name', 'Date_Received', 'Sales_Volume']):
            st.subheader("🔮 Forecast: Predict Future Sales for a Product")

            product_list = df['Product_Name'].dropna().unique()
            selected_product = st.selectbox("Select Product to Forecast", product_list)

            product_df = df[df['Product_Name'] == selected_product].copy()
            product_df['Date_Received'] = pd.to_datetime(product_df['Date_Received'], errors='coerce')
            product_df = product_df.dropna(subset=['Date_Received', 'Sales_Volume'])

            if not product_df.empty:
                product_df['Days_Since_Received'] = (datetime.today() - product_df['Date_Received']).dt.days
                forecast_df = product_df[['Days_Since_Received', 'Sales_Volume']].dropna()

                if not forecast_df.empty:
                    X = forecast_df[['Days_Since_Received']]
                    y = forecast_df['Sales_Volume']
                    model = LinearRegression()
                    model.fit(X, y)
                    future_days = np.array([[30], [60], [90]])
                    predictions = model.predict(future_days)

                    forecast_data = pd.DataFrame({
                        "Days Ahead": [30, 60, 90],
                        "Predicted Sales": predictions
                    })

                    fig_forecast = px.bar(forecast_data, x="Days Ahead", y="Predicted Sales",
                                          title=f"Predicted Sales for '{selected_product}' in Coming Days",
                                          text_auto='.2f')
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    for d, p in zip([30, 60, 90], predictions):
                        st.write(f"📦 In {d} days: **{p:.2f}** units expected for **{selected_product}**")
                else:
                    st.warning("⚠️ Not enough data to forecast this product.")
            else:
                st.warning("⚠️ Selected product has missing or invalid date/sales data.")
    else:
        st.info("📂 Please upload inventory data to generate reports.")





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

