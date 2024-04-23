import sys
import os 
import plotly.express as px
import streamlit as st 
import pandas as pd
from models import GenModel, helloworld
from data_ingestion import DataIngestion
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Home import Ingest_Data, plot_target_info, my_model
from logger import logging
from utils import load_object,save_objects
# Path to the artifact folder



def initialize_artifact_folder(folder_name):
    """
    Initialize the artifact folder, creating it if it doesn't exist.
    """
    if not os.path.exists(folder_name):
        st.info(f"The artifact folder '{folder_name}' does not exist. Creating the folder...")
        os.makedirs(folder_name)
    else:
        st.success(f"The artifact folder '{folder_name}' already exists.")


def check_artifact_folder(folder_name):
    """
    Check if the artifact folder is empty and return the list of files.
    """
    artifact_files = os.listdir(folder_name)
    if not artifact_files:
        st.warning(f"The artifact folder '{folder_name}' is empty. Please load data from the initial page or upload data manually.")
    else:
        st.write("Files in the artifact folder:")
        st.write(artifact_files)
    return artifact_files


def load_and_display_file(file_path):
    """
    Load and display the content of the given file based on its extension.
    """
    if file_path.endswith(".csv"):  # Load and display CSV files
        df = pd.read_csv(file_path)
        st.write("Loaded DataFrame from CSV:")
        st.dataframe(df)

    elif file_path.endswith(".pkl"):  # Load and display .pkl files
        # Custom function to load objects from pickle files
        obj = load_object(file_path)
        st.write("Loaded object from .pkl:")
        st.write(obj)  # Display the object

    else:  # If it's not a recognized format
        st.warning(f"Unrecognized file format: {file_path}")


 

# Plot histogram for a feature
def plot_unique_values(df, feature):
    fig = px.histogram(df, x=feature, title=f"Distribution of {feature}",color=feature)
    st.plotly_chart(fig)

# Display missing values
def display_missing_values(df, feature):
    missing_percentage = (df[feature].isna().sum() / df.shape[0]) * 100  # Missing values as a percentage
    non_missing_percentage = 100 - missing_percentage
    data = {
    "Status": ["Missing", "Non-Missing"],
    "Percentage": [missing_percentage, non_missing_percentage],
}
    missing_df = pd.DataFrame(data)
    fig = px.bar(
    missing_df,
    x='Status',
    y='Percentage',
    title=f"Percentage of Missing vs. Non-Missing Values for {feature}",
    color='Status',
    color_discrete_map={"Missing": "red", "Non-Missing": "green"},
    
)
    st.plotly_chart(fig)


 
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return v

# Function to perform Chi-Square Test
def chi_square_test(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p
def plot_feature_info(data,feature,target,name,description):
    feature_type = data[feature].dtype
    st.subheader(f"Feature Type : {feature_type}")
    if feature_type == 'object':
        display_missing_values(data, feature)
        st.button('Impute')
        st.write('----------')
        plot_unique_values(data, feature)
        options = st.selectbox("Select",['One-HotEncode','LabelEncode','Drop'])
        st.write('----------')
        if "cramer_explanation" not in st.session_state:
            st.session_state["cramer_explanation"] = None

        if "chi2_explanation" not in st.session_state:
            st.session_state["chi2_explanation"] = None

        # Calculate Cramér's V
        v = cramers_v(data[feature], data[target])
        st.write(f"Cramér's V between {feature} and {target}: {v:.2f}")
        if st.button("Generate cramer's V test Explination"):
            cramer_explanation_text = my_model.generate_content(f'in the context if {name} Dataset with brief description {description} explain the Cramér s V value between {feature} and {target} that is equal {v:.2f} ')
            st.session_state["cramer_explanation"] = cramer_explanation_text.text
        # Calculate Spearman's correlation
        # Perform and display Chi-Square Test results
        chi2, p = chi_square_test(data[feature], data[target])
        st.write(f"Chi-Square statistic: {chi2:.2f}")
        st.write(f"P-value: {p:.2e}")

        if st.button('Generate chi2 test Explination'):
            chi2_corr_text = my_model.generate_content(f"in the context if {name} Dataset with brief description {description} explain Chi-Square test result between {feature} and {target} that is equal chi2{chi2:.2f} and p-value {p:.2e}")
            st.session_state["chi2_explanation"] = chi2_corr_text.text

        if st.session_state["cramer_explanation"]:
            st.subheader("Cramer's V test explanation :")
            st.write(st.session_state["pearson_explanation"])

        if st.session_state["chi2_explanation"]:
            st.subheader("Chi-Square statistic's Explanation :")
            st.write(st.session_state["chi2_explanation"])



    if feature_type in ['int64', 'float64']:
        
        display_missing_values(data, feature)
        st.button('Impute')
        st.write('----------')
        plot_unique_values(data, feature)
        #scatterplot
        fig = px.scatter(data, x=feature, y=target, title=f"Scatter plot of {feature} vs {target}")
        st.plotly_chart(fig)
        
        # Histogram
        fig = px.histogram(data, x=feature, nbins=30, title=f"Histogram of {feature}")
        st.plotly_chart(fig)
        st.button('Standradize')
        st.button('Normalize')
        st.write('----------')
        # Box plot
        fig = px.box(data, y=feature, title=f"Box Plot of {feature}")
        st.plotly_chart(fig)
        st.button('Remove Outliers')
        st.write('----------')
        # Calculate Pearson's correlation
        pearson_corr = data[feature].corr(data[target], method='pearson')
        st.subheader(f"Pearson's correlation between {feature} and {target}: {pearson_corr:.2f}")
        if "pearson_explanation" not in st.session_state:
            st.session_state["pearson_explanation"] = None

        if "spearman_explanation" not in st.session_state:
            st.session_state["spearman_explanation"] = None

        if st.button("Generate Pearson's test Explination"):
            pearson_corr_text = my_model.generate_content(f"in the context if {name} Dataset with brief description {description} explain the Pearson correlation value between {feature} and {target} that is equal {pearson_corr:.2f} ")
            st.session_state["pearson_explanation"] = pearson_corr_text.text
        # Calculate Spearman's correlation
        spearman_corr = data[feature].corr(data[target], method='spearman')         
        st.subheader(f"Spearman's correlation between {feature} and {target}: {spearman_corr:.2f}")
        if st.button('Generate spearman test Explination'):
            spearman_corr_text = my_model.generate_content(f"in the context if {name} Dataset with brief description {description} explain theSpearman's correlation value between {feature} and {target} that is equal {spearman_corr:.2f} ")
            st.session_state["spearman_explanation"] = spearman_corr_text.text

        if st.session_state["pearson_explanation"]:
            st.subheader("Pearson's Explanation :")
            st.write(st.session_state["pearson_explanation"])

        if st.session_state["spearman_explanation"]:
            st.subheader("Spearman's Explanation :")
            st.write(st.session_state["spearman_explanation"])    
        
        


def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config(page_title="Feature Analysis", page_icon="🔍", layout='wide')
    artifact_folder = "artifact"  # Change to your folder's name

    initialize_artifact_folder(artifact_folder)
    artifact_files = check_artifact_folder(artifact_folder)

    if artifact_files:
        # Load and display the first file based on its extension
        data_description = load_object(os.path.join(artifact_folder, artifact_files[1]))
        data = load_object(os.path.join(artifact_folder, artifact_files[0]))
        st.write("Data Description:", data_description)
        st.sidebar.subheader(f"Name : {data_description['name']}")
        st.sidebar.subheader(f"Target : {data_description['target']}")
        st.sidebar.subheader(f"Type : {data_description['Data_type']}")

    # Page navigation logic
    st.title("Data Analysis App")
    target = data_description['target']
    feature_type = st.sidebar.selectbox("Pick a feature type", ['Categorical','Numerical'])
    if feature_type == 'Categorical':
        features_type = ['object']
    else:
        features_type = ['int64', 'float64']
    selection_feature = [x for x in data.columns if data[x].dtype in features_type]  
    feature = st.sidebar.selectbox("Pick a feature", selection_feature)
    
    st.sidebar.subheader(f'Picked Feature : {feature}')
    plot_feature_info(data,feature,target,name=data_description['name'],description=data_description['Data_type'])

        

if __name__ == "__main__":
    main()