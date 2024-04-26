import streamlit as st
import replicate
import time
import sys
import os
sys.path.append(r'\Data_Analysis_App') 
from models import GenModel
from logger import logging
from exceptions import CustomException
 

st.set_page_config(page_title="🦙💬 Assistant")

@st.cache_resource
def load_Gemini_model():
    try:
        model = GenModel(Google_Api_key,'gemini-pro')
        model.load_model()
        return model.model
    except Exception as e:
        st.error("Error loading model")
        logging.info(f"Error loading model", e)

def generate_response(name, temperature, top_p, max_length ,prompt_input,messages):
    
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. do not start the sentance with assitant: '."
    for dict_message in messages:
        content = dict_message["content"]

        if content is not None:
            if dict_message["role"] == "user":
                string_dialogue += f"User: {content}\n\n"
            else:
                string_dialogue += f"Assistant: {content}\n\n"

    if name == 'Llama3-70B':
        try:
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
            llama_3='meta/meta-llama-3-70b-instruct'
            output = replicate.run(llama_3, 
                                input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                        "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
            result = list(output)
            final_output = "".join(result)
            return final_output
        except Exception as e:
            st.info("Error loading Llama2-13B model choose other model")
            logging.error(f"Error loading model", e) 
    
    elif name == 'Llama3-8B':
        try:
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
            llama_3='meta/meta-llama-3-8b-instruct'
            output = replicate.run(llama_3, 
                                input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                        "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
            result = list(output)
            final_output = "".join(result)
            return final_output
        except Exception as e:
            st.info("Error loading Llama2-13B model choose other model")
            logging.error(f"Error loading model", e) 
    
    elif name == 'Gemini-pro':
        prompt = f"{string_dialogue} {prompt_input} Assistant: " + f'temperature={temperature}, {top_p}=top_p, max_length={max_length}, repetition_penalty=1'
        try:
            model = load_Gemini_model()
            response = model.generate_content(prompt).text
            return response
        except Exception as e:
            st.info("Error with Gemini-pro model choose other model")
            logging.error(f"Error loading model", e)                 
        
def main():
    with st.sidebar:
        st.title('🦙💬 Assistant ')
        st.subheader('Models and parameters')
        model_name = st.selectbox('Choose a model', ['Gemini-pro','Llama3-8B','Llama3-70B',])
        temperature = st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.slider('max_length', min_value=32, max_value=128, value=120, step=8)
     
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = [{"role": "assistant", "content": 'Hello there! How can I help you today?'}]
        st.session_state.chat_history = []

    if 'messages' not in st.session_state:
        st.session_state.messages=[{"role": "assistant", "content": 'Hello there! How can I help you today?'}]
        st.session_state.chat_history = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])    

    if prompt := st.chat_input('How can I help you today? '):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant") :
            with st.spinner("Thinking..."):
                try:                   
                    response = generate_response(model_name, temperature, top_p, max_length,prompt_input=prompt,messages=st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.markdown(response)
                except Exception as e:
                    st.info("There was a problem generating content. Please try again in 2 minutes or change The Model.")
                    logging.error(f"Error generating content", e)



Replicate_Api_key = 'r8_En9hgUT0LeT3JAqzCGYGR8PrK9NOiVF12T4xe' 
Google_Api_key = 'AIzaSyAqvNZVUqtGV11jRewG06nEH9_NJzZmpjI'   
Google_Api_key = st.sidebar.text_input("Enter your Google API key",type='password') 
Replicate_Api_key =st.sidebar.text_input("Enter your Replicate API key",type='password') 
os.environ['REPLICATE_API_TOKEN'] = Replicate_Api_key 
 
if not Google_Api_key and not Replicate_Api_key:
    st.info("Please enter your Google API key and Replicate API key")
elif not Google_Api_key:
    st.info("Please enter your Google API key")  
elif not Replicate_Api_key:
    st.info("Please enter your Replicate API key")
else:
    main()    

 
 
                      
#main()        