# AI-Powered Insights with Streamlit

This project provides an interactive interface using **Streamlit** to analyze financial data, generate SQL queries dynamically using **OpenAI**, and visualize insights. It connects to a **PostgreSQL** database and supports data visualization using **Seaborn** and **Matplotlib**.

---

## 🚀 Features

1. **Dynamic SQL Generation**
   - Translates natural language questions into SQL queries.
   - Supports PostgreSQL databases.

2. **Data Visualization**
   - Provides automated and query-specific visualizations:
     - Bar charts, scatter plots, line graphs, heatmaps, and more.
   - Custom visualization recommendations using AI.

3. **Database Support**
   - Works with financial datasets containing cards, transactions, user demographics, and fraud labels.

4. **AI-Powered Insights**
   - Summarizes query outputs into human-readable insights.
   - Uses OpenAI's GPT-4 for intelligent analysis.

---

## 📂 Dataset Overview

### **1. Cards Data**
- Contains details about issued cards, including card brands, types, credit limits, and chip information.

### **2. Transactions Data**
- Tracks purchase transactions, including amounts, merchant details, and modes of transaction.

### **3. User Demographics**
- Includes age, income, credit scores, and geolocation data for customers.

### **4. Merchant Categories**
- Business type classifications based on **Merchant Category Codes (MCC)**.

### **5. Fraud Labels**
- Flags fraudulent transactions for machine learning or pattern analysis.

---

## 🛠️ Setup Instructions

### Prerequisites
Ensure you have the following installed:
- **Python 3.9+**
- **PostgreSQL**
- **Streamlit**

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install dependencies :
   ```bash
   pip install -r requirements.txt
3. Set up your PostgreSQL database:

    Update the **db_config** in the code with your database details.
    Ensure the required tables and data are present.
4. Run the Streamlit application:
    ```bash
    streamlit run app.py
