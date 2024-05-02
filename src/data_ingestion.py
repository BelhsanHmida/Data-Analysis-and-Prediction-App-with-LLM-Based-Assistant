import sys
#sys.path.append(r'C:\Users\hp\Desktop\DataAnalysisApp\Data_Analysis_App')
import os
from dataclasses import dataclass
import pickle
import pandas as pd

from utils import save_objects 
from logger import logging
from exceptions import CustomException
 
from models import GenModel

Api_key='AIzaSyAqvNZVUqtGV11jRewG06nEH9_NJzZmpjI'

gen_model = GenModel(Api_key,'gemini-pro')
gen_model.load_model()

@dataclass  
class DataIngestion:
    def __init__(self,df,data_description,target,Data_type,Data_name):
        path = os.path.join("artifact", f"{Data_name}.csv")
        self.data_path = path
        logging.info("Reading data")
        self.data=df
        self.name = Data_name
        self.data_type = Data_type
        self.target = target
        self.description = data_description
        self.features = [x for x in self.data.columns if x!=target]
        self.X = self.data[self.features]
        self.y = self.data[self.target]
        self.Cat_features=self.X.select_dtypes(include='object').columns
        self.Num_features=self.X.select_dtypes(include=['int64','float64']).columns
        
        logging.info("Data read successfully")
                     
        
    def generate_data_overview(self):
        try:
            logging.info("Generating column overview")
            data_descriptions = {'Feature':self.data.columns.tolist(),
                                           'Data Type':[],
                                           'Unique count':[],
                                           'Missing count':[],
            }
            for feature in self.data.columns:
                data_descriptions['Unique count'].append(self.data[feature].nunique())
                data_descriptions['Data Type'].append(self.data[feature].dtype)
                missing_count = (self.data[feature].isnull().sum() /self.data.shape[0] )* 100
                data_descriptions['Missing count'].append(f"{missing_count:.2f}%")
                
            self.overview=pd.DataFrame(data_descriptions)    

        except Exception as e:
            logging.error(f"Error generating column description: {e}")
            raise CustomException(f"Error generating column description", e)
    def save_data(self,data_type,path):
        try:
            logging.info("Saving data")
            description={'name':self.name,'target':self.target,'Data_type':self.data_type}
            description_path = os.path.join("artifact", f"{self.name}_description.pkl")
            save_objects(description_path,description) 
            save_objects(path,self.data) 
            logging.info("Data and data_description saved successfully")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise CustomException(f"Error saving data", e)
        
    

if __name__ == "__main__" :
    
    genertated_text = gen_model.model.generate_content(' what is 4+5') 
    print(genertated_text.text)
    DataIngestion()      
    logging.info("Data ingestion completed")
