import sys
from logger import logging
from exceptions import CustomException
 
import google.generativeai as genai
 

 
class helloworld():
    def __init__(self):
        print("Hello World")
class GenModel():
    def __init__ (self,api_key,model_name): 
        self.api_key = api_key
        self.model_name =model_name
        self.model = None
        logging.info("Model initiated successfully")
            
    def load_model(self):
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model {e}")
            raise CustomException(f"Error loading model", e)
    def generate_content(self,prompt_text):
        try:
            content = self.model.generate_text(prompt_text)
            return content
        except Exception as e:
            logging.error(f"Error generating content {e}")
            raise CustomException(f"Error generating content", e)   

if __name__ == "__main__":
    my_model = GenModel(api_key=,'gemini-pro')
    my_model.load_model()
    genertated_text = my_model.model.generate_content('what is the son of a king called?') 

    print(genertated_text.text)       
