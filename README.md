# SEPM_Inventory_Management
SEPM_Inventory_Management

🧠 Smart Inventory Management System
A powerful, intelligent inventory management web application built with 🐍 Python, Streamlit, Pandas, Plotly, and Scikit-Learn. It helps businesses track stock, manage suppliers, monitor sales, generate forecasts, and checkout items with invoice generation.

🚀 Features
✅ Inventory Dashboard
✅ Product & Supplier Management
✅ Interactive Reports & Visualizations
✅ Sales Forecasting (Linear Regression)
✅ Smart Alerts (Low Stock, Supplier Distribution)
✅ Reorder Heatmap & Top Selling Products
✅ Point of Sale (Cashier Mode)
✅ Invoice Generation with Download
✅ User Authentication (optional)
✅ Mobile-friendly UI via Streamlit

🖥️ Modules Overview
📦 Inventory
View all products with key metrics like stock quantity, reorder level, etc.

Filter and search by Category, Supplier, or Status

Edit or update product info (coming soon)

📈 Reports
Sales volume over time (line chart)

Top-selling products (bar chart)

Inventory distribution (pie chart)

Reorder heatmap by warehouse

Supplier-wise stock bar graph

Revenue estimation by product

🔮 Forecast sales for 30/60/90 days using linear regression

🛒 Cashier (Point of Sale)
Add products to cart

Quantity input + total price calculation

Grand total at checkout

✅ Generate a unique invoice with:

Invoice ID

Date & time

Grand Total

📥 Download invoice as CSV

🏗️ Tech Stack
Frontend: Streamlit

Backend: Python

Data: Pandas, NumPy

ML: scikit-learn (LinearRegression)

Charts: Plotly Express

UUID: For invoice generation

🔧 Installation
1. Clone the repo
    git clone https://github.com/SidddhantJain/SEPM_Inventory_Management.git
    cd SEPM_Inventory_Management

2. Install dependencies
    pip install -r requirements.txt

3. Run the app
    streamlit run app.py

📁 Required CSV Format
Ensure your inventory CSV file includes the following columns:

Product_Name

Category

Supplier_Name

Warehouse_Location

Status

Product_ID

Supplier_ID

Date_Received

Last_Order_Date

Expiration_Date

Stock_Quantity

Reorder_Level

Reorder_Quantity

Unit_Price

Sales_Volume

Inventory_Turnover_Rate
![image](https://github.com/user-attachments/assets/168d50fc-4f4b-4199-a29c-a407f28b1fb7)


📸 Screenshots

DashBoard 
![image](https://github.com/user-attachments/assets/61541a01-0653-40a5-b4c0-c0b7fe7c8456)

Inventory View
![image](https://github.com/user-attachments/assets/721f8f01-88e9-456e-add4-fc015e92d903)

Forecast Charts
1) ![image](https://github.com/user-attachments/assets/05720e93-5476-43e8-953c-49f427c698a8)

2) ![image](https://github.com/user-attachments/assets/6289c5c5-6066-44f3-9f1e-f3a3352e90f4)

3) ![image](https://github.com/user-attachments/assets/65175bf5-fa1b-49db-850f-66046bc87387)

4) ![image](https://github.com/user-attachments/assets/823c3dbf-9189-4050-aa43-cd9cba91495d)


Cashier Checkout
![image](https://github.com/user-attachments/assets/79ca2929-7c34-4b5a-9004-8a13cc878c9d)

Invoice Download
![image](https://github.com/user-attachments/assets/c2cf235e-bcaa-4868-9e79-e876fa6c69eb)


(Comming Soon) 
🛠️ To-Do
 Edit inventory items directly from the app

 Role-based authentication (admin vs cashier)

 Generate printable PDF invoices

 Email invoice feature

 Real-time dashboard alerts

🤝 Contribution
Feel free to fork, improve, and create pull requests!
Ideas and suggestions are welcome.

🧑‍💻 Author
Built with ❤️ by Sidddhant Jain
🔗 GitHub: @SidddhantJain  



Link : https://sepminventorymanagement-6qhk2786vmmgy74y75ufuy.streamlit.app/
