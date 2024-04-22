import streamlit as st 
import pandas as pd
import plotly.express as px
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import google.generativeai as genai

from models import GenModel
from data_ingestion import DataIngestion


my_model = GenModel('AIzaSyAqvNZVUqtGV11jRewG06nEH9_NJzZmpjI','gemini-pro')
my_model.load_model()
my_model=my_model.model

@st.cache_data
def Ingest_Data(df, data_description, target, Data_type):
    data = DataIngestion(df, target)
    
    data.generate_column_description(data_description)
    
    return data

def plot_target_info(df):
    # Ensure 'target' is set in df and refers to a column name
    target=df.target
    df = df.data
    
    
    target_column = df[target]
    target_type = target_column.dtype

    st.subheader(f"Analysis for Target Column: {target_column}")
    st.write('target_type')
    # If the target is categorical, create a count plot and a pie chart
    if target_type == 'object':
        # Count plot
        count_plot = px.histogram(target_column, x=target_column, title=f"Count Plot for {target_column}")
        st.plotly_chart(count_plot)

        # Pie chart
        value_counts = target_column.value_counts()
        pie_chart = px.pie(value_counts.reset_index(), values=target_column, names='index', title=f"Proportions for {target_column}")
        st.plotly_chart(pie_chart)
    
    # If the target is numerical, create distribution plot and provide summary statistics
    elif target_type in ['int64', 'float64']:
        # Distribution plot
        distribution_plot = px.histogram(target_column, x=target_column, nbins=30, title=f"Distribution of {target_column}")
        st.plotly_chart(distribution_plot)
        
        # Summary statistics
        min_val = np.min(target_column)
        max_val = np.max(target_column)
        mean_val = np.mean(target_column)
        median_val = np.median(target_column)

        st.write(f"Summary Statistics for {target_column}:")
        st.write(f" - Min: {min_val}")
        st.write(f" - Max: {max_val}")
        st.write(f" - Mean: {mean_val}")
        st.write(f" - Median: {median_val}")

def plot_Num_data_info(df):
    Num_data=df.data[df.Num_features]
    # Correlation Analysis for Numerical Features
    st.subheader("Correlation Analysis for Numerical Features")
    correlation_matrix = Num_data.corr()
    fig_correlation = px.imshow(correlation_matrix, labels=dict(x="Features", y="Features", color="Correlation"))
    st.plotly_chart(fig_correlation)

def plot_Cat_data_info(df):
    Cat_data = df.data.select_dtypes(include='object')

    if Cat_data.empty:
        st.write("No categorical features found in the data.")
        return
    
    # Set a threshold to consider a feature for visualization
    unique_value_threshold = 10  # Adjust this threshold as needed

    st.subheader("Categorical Features Analysis")

    for feature in Cat_data.columns:
        unique_count = Cat_data[feature].nunique()

        if unique_count == 1:
            st.write(f"Feature: {feature} has only one unique value. Skipping visualization.")
            continue
        elif unique_count > unique_value_threshold:
            st.write(f"Feature: {feature} has too many unique values ({unique_count}). Showing top 10 value counts.")
            value_counts = Cat_data[feature].value_counts().head(10)  # Top 10 frequent values
        else:
            value_counts = Cat_data[feature].value_counts()

        st.write(f"Feature: {feature}")
        st.write(f"Unique values: {unique_count}")

        # Bar chart for unique value counts
        bar_fig = px.bar(value_counts, x=value_counts.index, y=value_counts.values,
                         title=f"Value Counts for {feature}",
                         labels={'x': feature, 'y': 'Count'})

        st.plotly_chart(bar_fig)

        # Pie chart for proportions of unique values
        pie_fig = px.pie(value_counts.reset_index(),
                         values='count',
                         names='index',
                         title=f"Proportions for {feature}")

        st.plotly_chart(pie_fig)

        st.write("-------------------------------------------------")

@st.cache_data
def plot_data_info(dataf,description):
    dataf.generate_column_description(description)
    description_df=dataf.description
    df=dataf.data

    st.subheader("Data Description")
    st.write(df.describe())
    st.subheader("Data Shape: ")
    st.write(f"Number of rows : {df.shape[0]} , Number of columns : {df.shape[1]}")

    st.subheader("Data columns")
     
    data=description.style.applymap(lambda x: 'color : green' ,subset=['Ai Generated description'])
 
    st.dataframe(data, width=1500)
    
    st.title("Numerical Features Analysis")
    plot_Num_data_info(dataf)
    
    st.title("Categorical Features Analysis")
    plot_Cat_data_info(dataf)
    st.title("Target Feature Analysis")
    plot_target_info(dataf)    







def reset_session_state():   
    for key in list(st.session_state.keys()):
        del st.session_state[key]
def main ():
    st.set_page_config(page_title="Data Analysis", page_icon="📊", layout='wide')
     
    col1, col2 = st.columns([6, 1]) 
    global data_name
    st.title("🤖 Data Exploration")
    with col2:
        st.write("")   
        if st.button("Reset", key="reset_button"):
            reset_session_state()
    st.title("Configuration")
    uploaded_files = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"], accept_multiple_files=True)
    
    with st.sidebar:
        last_selected_file_name = st.session_state.get("last_selected_file_name")

        if uploaded_files:
            file_names = [file.name for file in uploaded_files]
            selected_file_name = st.selectbox(
        "Select a CSV file for analysis",
        file_names,
        index=file_names.index(last_selected_file_name) if last_selected_file_name in file_names else 0,
    )
            st.session_state["last_selected_file_name"] = selected_file_name

            # If a file has been selected, load it and store it in session state
            if selected_file_name and (selected_file_name != last_selected_file_name or "selected_df" not in st.session_state):
                selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
                st.session_state["selected_df"] = pd.read_csv(selected_file)

    if "selected_df" in st.session_state :
        df   = st.session_state["selected_df"]
        name = st.session_state["last_selected_file_name"]
        st.subheader(f"Selected Dataset: {name}" )
        Data_type = st.radio('Select the type of Problem', ['Regression', 'Classification','Time-Series'])
        st.write("Data Type:", Data_type)
        data_description = st.text_input("Enter the Brief description of the data e.g. 'House price Dataset' ")
        st.write("Data Description:", data_description)
        target = st.selectbox("Select Target Feature", df.columns)
        st.write("Target Feature:", target)
        
    
    if "show_data" not in st.session_state:
        st.session_state.show_data = False 
    if st.button("Explore Data"):
        st.session_state.show_data = True
        st.text(f'data_description: {data_description}, target: {target}, Data Type: {Data_type}')
        df=Ingest_Data(df,data_description=data_description, target=target, Data_type=Data_type)
         
        
        plot_data_info(df,df.description)

    if st.session_state.show_data:
        #user_input = st.text_input("Enter something:", key="user_input")  
        if st.button("Submit"):
            st.write("You entered:", st.session_state.user_input)    


if __name__ == "__main__":  
    main()
