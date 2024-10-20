!pip install matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit_authenticator as stauth
from io import BytesIO
import base64

# Set page config
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")

# Dark mode theme
dark_theme_css = """
    <style>
    body {
        background-color: #0E1117; 
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #1f1f1f;
    }
    .css-1d391kg button {
        background-color: #303030;
        color: #ffffff;
    }
    .css-1d391kg input {
        background-color: #303030;
        color: #ffffff;
    }
    .css-1d391kg select {
        background-color: #303030;
        color: #ffffff;
    }
    </style>
    """
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Tabs to organize the app
tabs = st.tabs(["Upload Data", "Data Analysis", "Visualizations", "Modeling", "Reports"])

# Function to calculate descriptive statistics
def calculate_statistics(data):
    return data.describe()

# 1. File Upload
with tabs[0]:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV, Excel or TXT", type=["csv", "xlsx", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, delimiter='\t')

        st.write("### Dataset Preview", df.head())
        st.write("### Descriptive Statistics", calculate_statistics(df))

# 2. Data Analysis
with tabs[1]:
    st.header("Data Analysis & Filters")
    
    # Display column options for filtering
    if uploaded_file:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

        # Filter data with sliders, checkboxes, or text inputs
        st.sidebar.subheader("Filter Data")
        filter_column = st.sidebar.selectbox("Select a column to filter", numeric_columns + categorical_columns)
        if df[filter_column].dtype == np.number:
            min_value, max_value = st.sidebar.slider(f"Select range for {filter_column}", 
                                                     min_value=float(df[filter_column].min()), 
                                                     max_value=float(df[filter_column].max()))
            df_filtered = df[(df[filter_column] >= min_value) & (df[filter_column] <= max_value)]
        else:
            unique_values = df[filter_column].unique()
            selected_values = st.sidebar.multiselect(f"Select values for {filter_column}", unique_values, default=unique_values)
            df_filtered = df[df[filter_column].isin(selected_values)]
        
        st.write("### Filtered Data", df_filtered)

# 3. Visualizations
with tabs[2]:
    st.header("Visualize Data")
    
    # Dropdown to select type of plot
    plot_type = st.selectbox("Select plot type", ["Bar Chart", "Line Plot", "Scatter Plot", "Box Plot", "Histogram"])
    
    if uploaded_file:
        x_axis = st.selectbox("X-Axis", numeric_columns)
        y_axis = st.selectbox("Y-Axis", numeric_columns)
        
        if plot_type == "Bar Chart":
            fig = px.bar(df_filtered, x=x_axis, y=y_axis)
        elif plot_type == "Line Plot":
            fig = px.line(df_filtered, x=x_axis, y=y_axis)
        elif plot_type == "Scatter Plot":
            fig = px.scatter(df_filtered, x=x_axis, y=y_axis)
        elif plot_type == "Box Plot":
            fig = px.box(df_filtered, x=x_axis, y=y_axis)
        elif plot_type == "Histogram":
            fig = px.histogram(df_filtered, x=x_axis)
        
        st.plotly_chart(fig)

# 4. Model Selection
with tabs[3]:
    st.header("Modeling & Prediction")
    
    if uploaded_file:
        model_type = st.selectbox("Select Model", ["Linear Regression", "Logistic Regression"])
        if model_type == "Linear Regression":
            # Perform Linear Regression
            st.subheader("Linear Regression")
            features = st.multiselect("Select Features (X)", numeric_columns)
            target = st.selectbox("Select Target (Y)", numeric_columns)
            
            if st.button("Run Linear Regression"):
                X = df_filtered[features]
                Y = df_filtered[target]
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
                model = LinearRegression()
                model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
                
                st.write(f"### Model Coefficients: {model.coef_}")
                st.write(f"### Model Intercept: {model.intercept_}")
                st.write(f"### Predictions", predictions)
                
# 5. Generate Reports
with tabs[4]:
    st.header("Generate Report")
    
    if uploaded_file:
        report_button = st.button("Generate PDF Report")
        
        if report_button:
            buffer = BytesIO()
            # Create PDF (for simplicity, assuming we use text report, replace with real PDF generation in actual use)
            report_text = f"Data Report for {uploaded_file.name}\n\n"
            report_text += calculate_statistics(df).to_string()
            
            buffer.write(report_text.encode())
            buffer.seek(0)
            
            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:file/pdf;base64,{b64}" download="report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
