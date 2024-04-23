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
import re
from models import GenModel
from data_ingestion import DataIngestion
import shutil
from exceptions import CustomException
from logger import logging

my_model = GenModel(api_key=,'gemini-pro')
my_model.load_model()
my_model=my_model.model

@st.cache_data
def Ingest_Data(df, data_description, target, Data_type,Data_name):
    data = DataIngestion(df,data_description, target, Data_type,Data_name)
    
    data.generate_data_overview()
    
    return data

def plot_target_info(dataf):
    # Ensure 'target' is set in df and refers to a column name
    target=dataf.target
    df = dataf.data
    
    
    target_column = df[target]
    target_type = target_column.dtype

    st.subheader(f"Analysis for Target Column: {target}")
    st.write('Target type: ', target_type)
    # If the target is categorical, create a count plot and a pie chart
    if target_type == 'object':
        # Count plot
        count_plot = px.histogram(target_column, x=target_column, title=f"Count Plot for {target_column}")
        st.plotly_chart(count_plot)

        # Pie chart
        value_counts = target_column.value_counts()
        pie_chart = px.pie(value_counts.reset_index(), values=target_column, names='index', title=f"Proportions for {target}")
        st.plotly_chart(pie_chart)
    
    # If the target is numerical, create distribution plot and provide summary statistics
    elif target_type in ['int64', 'float64']:
        # Histogram
        distribution_plot = px.histogram(target_column, nbins=30, title=f"Distribution of {target}")
        st.plotly_chart(distribution_plot, use_container_width=True)

        # Box Plot
        box_plot = px.box(target_column, title=f"Box Plot of {target}")
        st.plotly_chart(box_plot, use_container_width=True)

        # Summary statistics
        min_val = np.min(target_column)
        max_val = np.max(target_column)
        mean_val = np.mean(target_column)
        median_val = np.median(target_column)

        st.subheader(f"Summary Statistics for {target}:")
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

def get_missing_color(missing_percentage):
    red_color_1 = '#ff0000'  # Red color Dim
    red_color_2 = '#ff0000'  # Red color Brighter
    yellow_color = '#ffff00'  # Yellow color
    green_color_1 = '#00ff00' # Green color Dim
    green_color_2 = '#00ff00' # Green color Brighter
    missing_percentage = float(missing_percentage.replace('%', ''))
    if 0 <= missing_percentage < 20:
        return f'background-color: {green_color_2}; color: black'
    elif 20<= missing_percentage < 30:
        return f'background-color: {green_color_1}; color: black'
    elif 30 <= missing_percentage < 50:
        return f'background-color: {yellow_color}; color: black'
    elif 50 <= missing_percentage <= 70:
        return f'background-color: {red_color_1}; color: black'
    elif missing_percentage > 70:
        return f'background-color: {red_color_2}; color: black'
    else:
        return "gray"
    
def color_green(val):
        return 'color: green'    
@st.cache_data
def plt_Feature_description (dataf):
    context = dataf.description
    features = dataf.data.columns
    descriptions={}
    for feature in features:
        description = my_model.generate_content(f"in the context of {context} dataset describe what is the feature :{feature} in 5 words or less")
        descriptions[feature]=description.text
    descriptions_df = pd.DataFrame(descriptions.items(), columns=['Feature', 'Ai Generated Description'])

    # Apply the lambda function to the specific column
    styled_df = descriptions_df.style.applymap(color_green, subset=['Ai Generated Description'])   
    st.dataframe(styled_df,width= 1500)    

def plot_data_info(dataf,description):
    dataf.generate_data_overview()
    description_df=dataf.overview
    df=dataf.data

    st.subheader("Data Description")
    st.write(df.describe())
    st.subheader("Data Shape: ")
    st.write(f"Number of rows : {df.shape[0]} , Number of columns : {df.shape[1]}")

    st.subheader("Data Overview :")
     
    data=description_df.style.applymap(get_missing_color, subset=['Missing count'])
    st.dataframe(data)
    
    st.subheader('Do you need help to understand Features?')
    try:
        plt_Feature_description(dataf)
    except Exception as e:
        st.warning("Could not load description info could be a token limit try again in 2 mins.")  # Display a warning message
        
        # You can also log the error or take other actions here
         

    st.title("Numerical Features Analysis")
    plot_Num_data_info(dataf)
    
    st.title("Target Feature Analysis :")
    plot_target_info(dataf)    


def add_space(space):
    for i in range(space):
        st.write("")

def reset_session_state():   
    for key in list(st.session_state.keys()):
        del st.session_state[key]
        

def clear_artifacts(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete the directory and its contents
        return True
    else:
        return False 
    
def validate_file_name(raw_name):
    # Disallowed characters that should not be in a file name
    forbidden_chars = r'[\ / ? % * : | " < >]'
    
    # Replace spaces with underscores and remove forbidden characters
    formatted_name = re.sub(forbidden_chars, "", raw_name.strip().replace(" ", "_"))

    return formatted_name

def main ():
    st.set_page_config(page_title="Data Analysis", page_icon="📊", layout='wide')
     
    col1, col2 = st.columns([6, 1]) 
    global data_name
    st.title("🤖 Data Exploration")
    with col2:
        st.write("")   
        if st.button("Reset", key="reset_button"):
            artifact_folder = "artifact" 
            if clear_artifacts(artifact_folder):
                st.success("Artifact folder has been cleared.")
            else:
                st.warning("Artifact folder does not exist or is already empty.")
            reset_session_state()
            st.experimental_rerun()
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
        st.write(df.head(20))
        st.write("------")
        add_space(1)
        
        st.subheader('Enter Data Name')
        pre_data_name = st.text_input("e.g HousePriceData")
        data_name = validate_file_name(pre_data_name)  
        if pre_data_name.strip().replace(" ", "_") == data_name:
             st.success(f"Valid file name: '{data_name}'")
        else:
            st.warning(
        f"The file name '{pre_data_name}' contained invalid characters or extra spaces. It has been formatted to '{data_name}'."
    )
        add_space(1)
        st.subheader("What type of problem are you trying to solve?")
        Data_type = st.radio('Select the type of Problem', ['Regression', 'Classification'])
        st.write("Data Type:", Data_type)
        add_space(1)
        
        st.subheader('Give a brief description of the data')
        data_description = st.text_input(" e.g. 'House price prediction Dataset' 'Alzheimer Medical tests Dataset ")
        st.write("Data Description:", data_description)
        
        add_space(1)
        reversed_columns = df.columns[::-1]
        target = st.selectbox("Select Target Feature", reversed_columns)
        st.write("Target Feature:", target)
        st.divider()

     

    if "Load Data" not in st.session_state:
        st.session_state.Load_data = False 

    if st.button("Load Data")  :
        st.session_state.Load_data = True
        st.text(f'data_description: {data_description}, target: {target}, Data Type: {Data_type}')
        dataf=Ingest_Data(df,data_description=data_description, target=target, Data_type=Data_type,Data_name=data_name)
        plot_data_info(dataf,data_description)
        st.write('--------')
        dataf.data.save_data(Data_type)
          
        col1, col2 = st.columns([1,2.3]) 
        with col2:
            st.subheader("Data Loaded Successfully!")


       
           

     


if __name__ == "__main__":  
    main()
