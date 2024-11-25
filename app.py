import streamlit as st
import openai
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_openai():
    """Initialize OpenAI API key from environment or user input"""
    if 'openai_api_key' not in st.session_state:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.session_state.openai_api_key = api_key
            openai.api_key = api_key
            return True
        return False
    openai.api_key = st.session_state.openai_api_key
    return True

# Dynamic Schema Retrieval for PostgreSQL
def fetch_schema():
    schema = '''
    Below are the tables along with the columns and their descriptions:
    
    Table: cards_data
    Columns:
    id (int): Unique identifier for the record. (ex: 1, 2, 3)
    client_id (int): Identifier for the client associated with the card. (ex: 101, 102)
    card_brand (text): Brand of the card (e.g., Visa, MasterCard). (ex: Visa, MasterCard)
    card_type (text): Type of card (e.g., Debit, Credit). (ex: Debit, Credit)
    card_number (text): Encrypted or masked card number for security. (ex: **** **** **** 1234)
    expires (date): Expiration date of the card in MM/YYYY format. (ex: 12/2024, 01/2025)
    cvv (int): Encrypted card security code for internal processing. (ex: 123, 456)
    has_chip (text): Indicates if the card is chip-enabled (YES/NO). (ex: YES, NO)
    num_cards_issued (int): Number of cards issued under this client ID. (ex: 3, 5)
    credit_limit (int): Maximum credit limit for the card in currency format. (ex: $5000, $20000)

    Table: transactions_data
    Columns:
    id (int): Unique identifier for the transaction. (ex: 7475327, 7475333)
    date (timestamp): Timestamp of the transaction. (ex: 1/1/2010 0:01)
    client_id (int): Unique identifier for the client. (ex: 1556, 1807)
    card_id (int): Identifier for the card used in the transaction. (ex: 2972, 165)
    amount (float): Transaction amount, including refunds or charges. (ex: $77.00, $4.81)
    use_chip (text): Mode of transaction indicating chip usage (e.g., Swipe Transaction). (ex: Swipe, Chip)
    merchant_id (int): Unique identifier for the merchant. (ex: 59935, 20519)
    merchant_city (text): City of the merchant. (ex: Beulah, Bronx)
    merchant_state (text): State where the merchant is located. (ex: ND, NY)
    zip (text): ZIP code of the merchant. (ex: 58523, 10464)
    mcc (int): Merchant Category Code (MCC). (ex: 5499, 5942)
    errors (text): Notes or indicators of issues with the transaction, if any. (ex: NULL, "Invalid" for failed transactions)
    
    Table: users_data
    Columns:
    id (int): Unique identifier for the customer. (ex: 1, 2, 3)
    current_age (date): Current age of the customer in date format. (ex: 1990-01-15)
    retirement_age (int): Expected retirement age of the customer. (ex: 65)
    birth_year (int): Year of birth of the customer. (ex: 1985, 1992)
    birth_month (int): Month of birth, possibly in a currency format error (e.g., $11, $12). (ex: 5, 6)
    gender (text): Gender of the customer. (ex: Male, Female)
    address (text): Residential address of the customer. (ex: 123 Main St, 456 Oak St)
    latitude (float): Geographical latitude. (ex: 40.7128)
    longitude (float): Geographical longitude. (ex: -74.0060)
    per_capita_income (float): Average income per person in the household. (ex: $25000)
    yearly_income (float): Annual income of the customer. (ex: $50000)
    total_debt (float): Total outstanding debt. (ex: $10000)
    credit_score (int): Creditworthiness score of the customer. (ex: 750)
    num_credit_cards (int): Number of credit cards owned by the customer. (ex: 2, 3)
    
    Table: mcc_codes
    Columns:
    mcc_code (int): Merchant Category Code (MCC), unique identifier for business types. (ex: 1711, 3000)
    business_type (text): Description of the business category. (e.g., "Heating, Plumbing, Air Conditioning Contractors")

    Table: train_fraud_labels
    Columns:
    id (int): Unique identifier for each transaction. (ex: 7476734, 7481767)
    labels (text): Binary classification label indicating the outcome of the transaction (Yes/No). (ex: Yes, No)'''
    return schema

# AI SQL Query Generation
def generate_sql_query1(user_question, schema):
    
    prompt = f"""
    You are an expert SQL generator. Translate the following user question into a PostgreSQL SQL query.
    Database Schema:
    {schema}
    Question: {user_question}
    SQL Query:
    
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert generating PostgreSQL queries.Please note that if you generate any other comments apart from query please add it as a comment and also please don't include sql query in any sql query block"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
            
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Error generating SQL query: {e}")
        
def generate_sql_query(user_question, schema):
    prompt = f"""
    You are an expert SQL generator. Translate the following user question into a PostgreSQL SQL query.
    Database Schema:
    {schema}
    Question: {user_question}
    SQL Query:
    -- Please ensure that any comments in the query are written in PostgreSQL format (i.e., starting with '--').
    -- Do not include SQL query within a SQL query block.
    -- Avoid using any comments that are not related to the query itself.

    """
    try:
        # Send the request to OpenAI API for query generation
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert generating PostgreSQL queries. Please note that if you generate any other comments apart from the query itself, please add them as a comment and also ensure that comments are in proper PostgreSQL format (i.e., '--')."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
        )
        
        sql_query = response.choices[0].message.content.strip()

        # Post-process the generated SQL to ensure comments are in the correct format
        # and that no SQL query block is nested inside another query.
        
        # Check if there's any misplaced comment or nested SQL query block
        if '--' in sql_query:
            # Clean up comments if needed (ensure they are correctly placed in the query)
            sql_query = "\n".join(line.strip() for line in sql_query.splitlines())

        return sql_query
    except Exception as e:
        raise ValueError(f"Error generating SQL query: {e}")


# Execute SQL Query on PostgreSQL
def execute_query(sql_query, db_config):
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        raise ValueError(f"SQL execution error: {e}")

# Generate Insights from AI
def generate_insights(user_question, query_output_df):
    try:
        summary_prompt = f"""
        Data Preview:
        {query_output_df.head(10).to_csv(index=False)}
        User Question: {user_question}
        Generate concise insights based on the data and user question. Please maintain a unique font. 
        DONOT GIVE RANDOM FONTS AND DO NOT HIGHLIGHT ANYTHING.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst providing insights from data."},
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=300,
            temperature=0,
            
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise ValueError(f"Error generating insights: {e}")

# Visualization Function
def visualize_data1(query_output_df):
    try:
        # Automatically detect the most appropriate visualization based on columns
        num_columns = len(query_output_df.columns)

        if num_columns == 2:
            # If there's one categorical column and one numeric column, use a bar chart
            if query_output_df.dtypes[1] in ['int64', 'float64']:
                st.bar_chart(query_output_df.set_index(query_output_df.columns[0]))
            else:
                st.write("Unsupported data for bar chart visualization.")

        elif num_columns == 3:
            # If there's one categorical and two numeric columns, use a scatter plot or line chart
            if query_output_df.dtypes[1] in ['int64', 'float64'] and query_output_df.dtypes[2] in ['int64', 'float64']:
                fig, ax = plt.subplots()
                sns.scatterplot(
                    data=query_output_df,
                    x=query_output_df.columns[1],
                    y=query_output_df.columns[2],
                    hue=query_output_df.columns[0],
                    ax=ax
                )
                st.pyplot(fig)
            else:
                st.line_chart(query_output_df.set_index(query_output_df.columns[0]))

        elif num_columns == 4:
            # If there are three numeric columns, generate a scatter plot with size
            if all(query_output_df.dtypes[i] in ['int64', 'float64'] for i in range(1, 4)):
                fig, ax = plt.subplots()
                sns.scatterplot(
                    data=query_output_df,
                    x=query_output_df.columns[1],
                    y=query_output_df.columns[2],
                    size=query_output_df.columns[3],
                    hue=query_output_df.columns[0],
                    sizes=(20, 200),
                    ax=ax
                )
                st.pyplot(fig)

        elif num_columns > 4:
            # If there are many columns, consider a heatmap of correlations (if numeric)
            numeric_cols = query_output_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots()
                sns.heatmap(query_output_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.write("Too many columns for visualization. Consider filtering the data.")

        else:
            st.write("Unsupported data format for visualization.")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

def visualize_data2(query_output_df, user_question):
    try:
        # Preprocessing
        numeric_columns = query_output_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = query_output_df.select_dtypes(include=['object', 'category']).columns
        # Keyword-based visualization selection
        def select_visualization_by_query():
            query_lower = user_question.lower()
            # Distribution and trend-related keywords
            if any(word in query_lower for word in ['trend', 'distribution', 'spread', 'range']):
                if len(numeric_columns) > 1:
                    # Histogram or distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for col in numeric_columns:
                        sns.histplot(query_output_df[col], kde=True, ax=ax, label=col)
                    ax.set_title('Distribution of Numeric Variables')
                    ax.legend()
                    st.pyplot(fig)
                    return True
            # Comparison-related keywords
            if any(word in query_lower for word in ['compare', 'difference', 'contrast', 'vs', 'versus']):
                if len(categorical_columns) > 0 and len(numeric_columns) > 0:
                    # Box plot or violin plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cat_col = categorical_columns[0]
                    num_col = numeric_columns[0]
                    sns.boxplot(x=cat_col, y=num_col, data=query_output_df, ax=ax)
                    ax.set_title(f'{num_col} Comparison by {cat_col}')
                    st.pyplot(fig)
                    return True
            # Relationship-related keywords
            if any(word in query_lower for word in ['relationship', 'correlation', 'impact', 'effect']):
                if len(numeric_columns) > 1:
                    # Correlation heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = query_output_df[numeric_columns].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation Heatmap')
                    st.pyplot(fig)
                    return True
            return False
 
        # Default visualization based on data structure
        def default_visualization():
            # Scatter plot for 2-3 numeric columns
            if len(numeric_columns) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                if len(numeric_columns) == 2:
                    sns.scatterplot(x=numeric_columns[0], y=numeric_columns[1], data=query_output_df, ax=ax)
                else:
                    sns.scatterplot(x=numeric_columns[0], y=numeric_columns[1], 
                                    hue=numeric_columns[2], data=query_output_df, ax=ax)
                ax.set_title('Scatter Plot of Numeric Variables')
                st.pyplot(fig)
            # Bar plot for categorical vs numeric
            elif len(categorical_columns) > 0 and len(numeric_columns) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                cat_col = categorical_columns[0]
                num_col = numeric_columns[0]
                query_output_df.groupby(cat_col)[num_col].mean().plot(kind='bar', ax=ax)
                ax.set_title(f'Average {num_col} by {cat_col}')
                st.pyplot(fig)
            # Simple dataframe summary if no good visualization
            else:
                st.write("Data Summary:")
                st.dataframe(query_output_df)
 
        # Try query-based visualization first, fall back to default
        if not select_visualization_by_query():
            default_visualization()
 
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        # Fallback to displaying raw data
        st.dataframe(query_output_df)

def visualize_data(query_output_df, user_question):
    try:
        # Prepare data for AI analysis
        data_preview = query_output_df.head(10).to_csv(index=False)
        column_types = {col: str(dtype) for col, dtype in query_output_df.dtypes.items()}
        # Prompt for visualization recommendation
        visualization_prompt = f"""
        You are an expert data visualization strategist. Provide a DETAILED, STRUCTURED recommendation for data visualization.
        IMPORTANT: Respond STRICTLY in the following JSON format:
        {{
            "type": "visualization type (bar/scatter/line/box/heatmap/pie)",
            "x_axis": "column name for x-axis",
            "y_axis": "column name for y-axis",
            "hue": "column for color/grouping (optional)",
            "title": "Visualization title",
            "reasoning": "Brief explanation of visualization choice"
        }}
 
        User Question: {user_question}
        Data Preview (Top 10 Rows):
        {data_preview}
        Column Types:
        {column_types}
        Recommendation Guidelines:
        1. Analyze data types and user intent
        2. Choose most informative visualization
        3. Consider columns available
        4. Prioritize clarity and insights
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data visualization strategist. Always respond in strict JSON format."},
                    {"role": "user", "content": visualization_prompt},
                ],
                max_tokens=300,
                temperature=0,
               
            )
            # Extract and parse JSON from text response
            viz_recommendation_text = response.choices[0].message.content.strip()
            viz_recommendation = json.loads(viz_recommendation_text)
        except json.JSONDecodeError:
            st.error("Failed to parse AI response. Falling back to default visualization.")
            return default_visualization(query_output_df)
        except Exception as e:
            st.error(f"Error in AI visualization analysis: {e}")
            return default_visualization(query_output_df)
        # Create visualization based on recommendation
        plt.figure(figsize=(12, 6))
        # Visualization Type Mapping
        if viz_recommendation['type'] == 'bar':
            x_col = viz_recommendation['x_axis']
            y_col = viz_recommendation['y_axis']
            hue_col = viz_recommendation.get('hue')
            if hue_col and hue_col in query_output_df.columns:
                sns.barplot(x=x_col, y=y_col, hue=hue_col, data=query_output_df)
            else:
                sns.barplot(x=x_col, y=y_col, data=query_output_df)
        elif viz_recommendation['type'] == 'scatter':
            x_col = viz_recommendation['x_axis']
            y_col = viz_recommendation['y_axis']
            hue_col = viz_recommendation.get('hue')
            if hue_col and hue_col in query_output_df.columns:
                sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=query_output_df)
            else:
                sns.scatterplot(x=x_col, y=y_col, data=query_output_df)
        elif viz_recommendation['type'] == 'line':
            x_col = viz_recommendation['x_axis']
            y_col = viz_recommendation['y_axis']
            plt.plot(query_output_df[x_col], query_output_df[y_col])
        elif viz_recommendation['type'] == 'box':
            x_col = viz_recommendation['x_axis']
            y_col = viz_recommendation['y_axis']
            sns.boxplot(x=x_col, y=y_col, data=query_output_df)
        elif viz_recommendation['type'] == 'heatmap':
            # Correlation heatmap
            numeric_cols = query_output_df.select_dtypes(include=['int64', 'float64']).columns
            correlation_matrix = query_output_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        elif viz_recommendation['type'] == 'pie':
            x_col = viz_recommendation['x_axis']
            y_col = viz_recommendation['y_axis']
            plt.pie(query_output_df[y_col], labels=query_output_df[x_col], autopct='%1.1f%%')
        else:
            st.warning("Unsupported visualization type")
            return default_visualization(query_output_df)
        # Set title and labels
        plt.title(viz_recommendation.get('title', 'Data Visualization'))
        plt.xlabel(viz_recommendation.get('x_axis', 'X-Axis'))
        plt.ylabel(viz_recommendation.get('y_axis', 'Y-Axis'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Display visualization and reasoning
        st.pyplot(plt)
        st.markdown(f"**Visualization Reasoning:** {viz_recommendation.get('reasoning', 'No specific reasoning provided')}")
 
    except Exception as e:
        st.error(f"Comprehensive visualization error: {str(e)}")
        # Fallback to default visualization
        default_visualization(query_output_df)
 
def default_visualization(query_output_df):
    """Fallback visualization method"""
    try:
        numeric_cols = query_output_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=query_output_df, ax=ax)
            plt.title('Default Scatter Plot')
            st.pyplot(fig)
        else:
            st.dataframe(query_output_df)
    except Exception as e:
        st.error(f"Default visualization failed: {str(e)}")
        st.dataframe(query_output_df)
 
def get_db_config():
    return {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5431")
    }

# Streamlit Application
def main():
    st.set_page_config(layout="wide",page_title="AI Smart Insights") 
    
    if not init_openai():
        api_key_input = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            openai.api_key = api_key_input
        else:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
    
    
    st.title("AI Smart Insights - Based on TAG Architecture")
    st.sidebar.title("About Data")
    #username = st.sidebar.text_input("Enter your username:")
    st.sidebar.markdown("""
    ## Dataset Overview
    
    ### Data Components
    - **Cards Data**: Card details, brands, types
    - **Transactions**: Purchase records, merchant info
    - **User Demographics**: Age, income, credit profiles
    - **Merchant Categories**: Business type classifications
    - **Fraud Labels**: Fraudulent transactions flag
    
    
    ### Key Focus
    - Transaction pattern analysis
    - Customer behavior insights
    - Potential fraud detection
    
    ### Data Highlights
    - Multiple data sources interconnected
    - Comprehensive financial tracking    - Geospatial and demographic context
    """)
    user_question = st.text_input("Ask your database question:")

    # PostgreSQL Configuration
    db_config = get_db_config()  
   

    if not user_question:
        st.sidebar.warning("Please provide the question on the data.")
        st.stop()

    # Fetch Schema
    try:
        schema = fetch_schema()
    except Exception as e:
        st.error(f"Error fetching database schema: {e}")
        st.stop()

    # Generate SQL Query
    try:
        sql_query = generate_sql_query(user_question, schema)
    except ValueError as e:
        st.error(e)
        st.stop()

    # Execute Query
    try:
        query_result = execute_query(sql_query, db_config)
    except ValueError as e:
        st.error(e)
        st.stop()

    # Layout
    col1, col2 = st.columns([4,4])

    # Column 1: SQL Query and Results
    with col1:
        st.subheader("Generated SQL Query")
        st.code(sql_query, language="sql")
        st.subheader("Query Results")
        st.dataframe(query_result)

    # Column 2: Insights and Visualization
    with col2:
        st.subheader("AI-Generated Insights")
        try:
            insights = generate_insights(user_question, query_result)
            st.write(insights)
        except ValueError as e:
            st.error(e)

        st.subheader("Visualization")
        visualize_data(query_result,user_question)

if __name__ == "__main__":
    main()
