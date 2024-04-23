import sys
from logger import logging
from exceptions import CustomException

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
    
if __name__ == "__main__":
    file_path = f"artifact/{'hello'}test.pkl"
    file_path_csv = f"artifact/{'hello'}test.csv"    
    save_objects(file_path,np.array([1,2,3,4,5])) 
    save_objects(file_path_csv,pd.DataFrame({'A':[1,2,3,4,5],'B':[6,7,8,9,10]})) 
    print(load_object(file_path))
    print(load_object(file_path_csv))
