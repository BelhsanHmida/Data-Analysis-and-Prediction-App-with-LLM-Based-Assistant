import sys
sys.path.append(r'C:\Users\hp\Desktop\DataAnalysisApp\Data_Analysis_App')
from logger import logging
from exceptions import CustomException

import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
import pickl
import dill
 

def save_objects(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e :
        raise CustomException(e,sys)    
    
def load_object(file_path) :
    try:
        with open(file_path,"rb")as file_obj:
            return dill.load(file_obj)
    except Exception as e :
       raise CustomException(e,sys)
    