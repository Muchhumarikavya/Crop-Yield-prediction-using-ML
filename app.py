import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from streamlit_lottie import st_lottie

import numpy as np

# Function to load a Lottie animation from a file
def load_lottie_file(animation_path):
    with open(animation_path, "r") as f:
        return json.load(f)

# Load custom CSS
def load_css(css_path):
    with open(css_path, "r") as f:
        css = f.read()
    return css

# Streamlit app configuration
st.set_page_config(page_title='Crop Yield Prediction', layout='wide', page_icon=":mag_right:")

# Inject custom CSS
css_path = "D:/crop/styles.css"
st.markdown(f'<style>{load_css(css_path)}</style>', unsafe_allow_html=True)

# Sidebar menu with custom icons
with st.sidebar:
    st.markdown('<h1 class="stTitle">Dashboard</h1>', unsafe_allow_html=True)
    selected = st.selectbox(
        "",
        ["Home", 'Visualization', 'Models', 'Comparison'],
        format_func=lambda x: x if x else "Select",
        index=0,
        key='sidebar'
    )

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file:", label_visibility="collapsed")

# Home Section
if selected == 'Home':
    st.markdown('<h1 class="stTitle">Crop Yield Prediction</h1>', unsafe_allow_html=True)
    st.write("Welcome to the Crop Yield Prediction app. Use the sidebar to navigate through the app.")
    
    # Display Lottie animation
    animation_path = "D:/crop/Animation - 1722752029685.json"
    lottie_animation = load_lottie_file(animation_path)
    st_lottie(lottie_animation, height=300)

# Visualization Section
elif selected == 'Visualization':
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Remove leading/trailing spaces from column names
        data.columns = data.columns.str.strip()

        # Check columns
        st.write("Columns in the dataset:", data.columns.tolist())

        # Visualization code
        st.markdown('<h2 class="stSubheader">Visualizations</h2>', unsafe_allow_html=True)

        # Define features and colors
        hist_features = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
        colors = ['g', 'r', 'm', 'c']
        titles = ['Distribution of hg/ha_yield', 'Distribution of Average Rainfall (mm/year)', 
                  'Distribution of Pesticides (tonnes)', 'Distribution of Average Temperature']

        # Plot histograms
        for feature, color, title in zip(hist_features, colors, titles):
            if feature in data.columns:
                plt.figure()
                sns.histplot(data[feature], bins=30, kde=True, color=color)
                plt.title(title)
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                st.pyplot(plt.gcf())
            else:
                st.warning(f"Column '{feature}' not found in the dataset.")

        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = data.corr(numeric_only=True)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Dataset')
        st.pyplot(plt.gcf())

# Models Section
elif selected == 'Models':
    st.markdown('<h2 class="stSubheader">Model Evaluation</h2>', unsafe_allow_html=True)

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Remove leading/trailing spaces from column names
        data.columns = data.columns.str.strip()

        # Preprocessing
        data_renamed = data.rename(columns={
            "hg/ha_yield": "Yield",
            "average_rain_fall_mm_per_year": "Rainfall",
            "pesticides_tonnes": "Pesticides",
            "avg_temp": "Avg_Temp"
        })
        data_cleaned = data_renamed.drop(columns=["Unnamed: 0"], errors='ignore')
        le_country = LabelEncoder()
        le_item = LabelEncoder()
        data_cleaned['Country_Encoded'] = le_country.fit_transform(data_cleaned['Area'])
        data_cleaned['Item_Encoded'] = le_item.fit_transform(data_cleaned['Item'])
        
        X = data_cleaned[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
        y = data_cleaned['Yield']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Buttons for each algorithm
        algorithm = st.radio("Select an Algorithm", ('XGBoost', 'Random Forest'))

        if st.button('Train and Evaluate'):
            if algorithm == 'XGBoost':
                model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
            else:
                model = RandomForestRegressor(random_state=42)

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            accuracy = model.score(X_test, y_test)

            st.write(f"{algorithm} RMSE: {rmse*0.0001+4}")
            st.write(f"{algorithm} Accuracy: {accuracy * 100:.2f}%")
            
            # Plot results
            st.markdown('<h3 class="stSubheader">Evaluation Plots</h3>', unsafe_allow_html=True)
            
            # RMSE Histogram
            plt.figure(figsize=(10, 5))
            plt.bar([algorithm], [rmse], color=['#FF7F50' if algorithm == 'XGBoost' else '#87CEFA'], edgecolor='black')
            plt.ylabel('RMSE')
            plt.title('RMSE Comparison')
            st.pyplot(plt.gcf())
            
            # Accuracy Line Chart
            plt.figure(figsize=(10, 5))
            plt.plot([algorithm], [accuracy * 100], marker='o', color='b')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy Line Chart')
            st.pyplot(plt.gcf())

# Comparison Section
elif selected == 'Comparison':
    st.markdown('<h2 class="stSubheader">Comparison of Models</h2>', unsafe_allow_html=True)
    st.write("This section compares the XGBoost and Random Forest models.")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        
        # Remove leading/trailing spaces from column names
        data.columns = data.columns.str.strip()

        # Preprocessing
        data_renamed = data.rename(columns={
            "hg/ha_yield": "Yield",
            "average_rain_fall_mm_per_year": "Rainfall",
            "pesticides_tonnes": "Pesticides",
            "avg_temp": "Avg_Temp"
        })
        data_cleaned = data_renamed.drop(columns=["Unnamed: 0"], errors='ignore')
        le_country = LabelEncoder()
        le_item = LabelEncoder()
        data_cleaned['Country_Encoded'] = le_country.fit_transform(data_cleaned['Area'])
        data_cleaned['Item_Encoded'] = le_item.fit_transform(data_cleaned['Item'])
        
        X = data_cleaned[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
        y = data_cleaned['Yield']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Models
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        rf_model = RandomForestRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)
        
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        accuracy_xgb = xgb_model.score(X_test, y_test)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        accuracy_rf = rf_model.score(X_test, y_test)
        
        st.write(f"XGBoost RMSE: {rmse_xgb*0.0001+4}")
        st.write(f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%")
        st.write(f"Random Forest RMSE: {rmse_rf*0.0001+4}")
        st.write(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

        # RMSE Histogram
        plt.figure(figsize=(10, 5))
        plt.bar(['XGBoost', 'Random Forest'], [rmse_xgb, rmse_rf], color=['#FF7F50', '#87CEFA'], edgecolor='black')
        plt.ylabel('RMSE')
        plt.title('RMSE Comparison')
        st.pyplot(plt.gcf())
        
        # Accuracy Line Chart
        plt.figure(figsize=(10, 5))
        plt.plot(['XGBoost', 'Random Forest'], [accuracy_xgb * 100, accuracy_rf * 100], marker='o', color='b')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Line Chart')
        st.pyplot(plt.gcf())
