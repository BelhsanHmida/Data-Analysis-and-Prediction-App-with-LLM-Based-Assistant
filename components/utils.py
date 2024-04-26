import sys
from logger import logging
from exceptions import CustomException
import streamlit as st
import shutil
import re
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
import pickl
import dill
 

def save_objects(file_path, obj):
    try:
        # Ensure the parent directory exists
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Determine the file extension
        file_extension = os.path.splitext(file_path)[1]

        if file_extension == ".csv":
            # Save as CSV
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(file_path, index=False)  # Write DataFrame to CSV
            else:
                with open(file_path, 'w') as file_obj:
                    file_obj.write(obj)  # Write other text-based data

        elif file_extension == ".pkl":
            # Save as Pickle/PKL
            with open(file_path, 'wb') as file_obj:  # Binary mode for PKL
                dill.dump(obj, file_obj)

        else:
            raise ValueError("Unsupported file extension")  # Handle other extensions

    except Exception as e:
        raise CustomException(e, sys)  # Handle the exception 
    

def load_object(file_path):
    try:
        # Determine the file extension
        file_extension = os.path.splitext(file_path)[1]

        if file_extension == ".csv":
            # Load CSV file
            return pd.read_csv(file_path)  # Returns a DataFrame

        elif file_extension == ".pkl":
            # Load PKL (Pickle) file
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)  # Returns the deserialized object

        else:
            raise ValueError("Unsupported file extension")  # Handle unexpected extensions

    except Exception as e:
        raise CustomException(e, sys) 



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
        6
    return artifact_files


def load_and_display_file(file_path):
    """
    Load and display the content of the given file based on its extension.
    """
    if file_path.endswith(".csv"):  # Load and display CSV files
        df = pd.read_csv(file_path)
        st.dataframe(df)

    elif file_path.endswith(".pkl"):  # Load and display .pkl files
        # Custom function to load objects from pickle files
        obj = load_object(file_path)
          # Display the object

    else:  # If it's not a recognized format
        st.warning(f"Unrecognized file format: {file_path}")





if __name__ == "__main__":
    file_path = f"artifact/{'hello'}test.pkl"
    file_path_csv = f"artifact/{'hello'}test.csv"    
    save_objects(file_path,np.array([1,2,3,4,5])) 
    save_objects(file_path_csv,pd.DataFrame({'A':[1,2,3,4,5],'B':[6,7,8,9,10]})) 
    print(load_object(file_path))
    print(load_object(file_path_csv))
