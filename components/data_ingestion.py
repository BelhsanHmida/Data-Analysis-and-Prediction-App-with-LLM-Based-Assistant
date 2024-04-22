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
'''
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)'''

 
 
gen_model = GenModel(API_key,'gemini-pro')
gen_model.load_model()


    
@dataclass  
class DataIngestion:
    def __init__(self,df,target):

        self.data_path = os.path.join("artifact", "data.csv")
        logging.info("Reading data")
        self.data=df
        self.target = target
        self.features = [x for x in self.data.columns if x!=target]
        self.X = self.data[self.features]
        self.y = self.data[self.target]
        self.Cat_features=self.X.select_dtypes(include='object').columns
        self.Num_features=self.X.select_dtypes(include=['int64','float64']).columns
        
        logging.info("Data read successfully")
                     
   
        
    def generate_column_description(self,data_description):
        try:
            logging.info("Generating column description")
            data_descriptions = {'Feature':self.data.columns.tolist(),
                                           'Data Type':[],
                                           'Missing count':[],
                                           'Ai Generated description':[]}
            for feature in self.data.columns:
                feature_description = gen_model.model.generate_content(f"Given the following data description ({data_description}) and the a specific feature name {feature}, provide a concise summary of the feature in 7 words or fewer.") 
                data_descriptions['Ai Generated description'].append(feature_description.text)
                data_descriptions['Data Type'].append(self.data[feature].dtype)
                missing_count = self.data[feature].isnull().mean() * 100
                data_descriptions['Missing count'].append(self.data[feature].isnull().sum())
                
            self.description=pd.DataFrame(data_descriptions)    

        except Exception as e:
            logging.error(f"Error generating column description: {e}")
            raise CustomException(f"Error generating column description", e)

if __name__ == "__main__" :
    
    genertated_text = gen_model.model.generate_content(' what is 4+5') 
    print(genertated_text.text)
    DataIngestion()      
    logging.info("Data ingestion completed")
