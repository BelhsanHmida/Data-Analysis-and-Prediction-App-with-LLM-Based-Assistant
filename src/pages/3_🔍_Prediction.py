import streamlit as st
import pandas as pd
import numpy as np

from io import StringIO 

import plotly.graph_objs as go
import plotly.express as px
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
 
 
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
 
from logger import logging
from utils import load_object,save_objects,initialize_artifact_folder,check_artifact_folder,get_api_key,load_Gemini_model

import os

 

Google_Api_key = get_api_key("Google_API_KEY")
 
Gemini_model=load_Gemini_model(Google_Api_key)


 

st.set_page_config(page_title="🔍 Prediction")
 
def load_pred_model(model_name):
    if model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'Random Forest': 
        model = RandomForestClassifier()
    elif model_name == 'XGBoost classifier':
        model = XGBClassifier()
    elif model_name == 'Gradient Boosting classification':
        model = GradientBoostingClassifier()
    elif model_name == 'Random Forest Regression':
        model = RandomForestRegressor()
    elif model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Ridge Regression':
        model = Ridge()
    elif model_name == 'Lasso Regression':
        model = Lasso()
    elif model_name == 'XGBoost regressor':
        model = XGBRegressor()
    elif model_name == 'Gradient Boosting Regression':
        model = GradientBoostingRegressor()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
 



def select_file(artifact_folder='artifact'):

    initialize_artifact_folder(artifact_folder)
    artifact_files = check_artifact_folder(artifact_folder)
    selection_list = [x for x in artifact_files if x.startswith("Modified_") and x.endswith('.csv') ]
    selected_data = st.sidebar.selectbox('Pick DataSet',selection_list)
    selected_data_description = selected_data.split('.')[0] + '_description.pkl'
    if artifact_files:
        # Load and display the first file based on its extension
        data_description = load_object(os.path.join(artifact_folder,selected_data_description))
        data = load_object(os.path.join(artifact_folder, selected_data))
        
        
    return data, data_description 

#main page    

if True:
    # Load and display the first file based on its extension
    data ,data_description = select_file(artifact_folder='artifact')
    st.write("Data Description:", data_description)
 
    st.sidebar.subheader(f"Target : {data_description['target']}")
    if data_description['Data_type'] == 'Classification':
        classification_type = st.sidebar.radio(
    "Choose Classification Type",
    ('Binary', 'Multi-class')
)
        if classification_type == 'Binary':
            average_type = 'binary'  # suitable for binary classification
        else:
            average_type = 'weighted'

        recomended_models = ['Logistic Regression', 'Random Forest','XGBoost classifier', 'Gradient Boosting classification']
    else:
        recomended_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression','XGBoost regressor', 'Random Forest Regression', 'Gradient Boosting Regression']      
    st.sidebar.subheader(f"Type : {data_description['Data_type']}")
    st.sidebar.subheader("Recommendend Models : ")

    Pred_model=st.sidebar.selectbox("Pick a model", recomended_models)
    if 'prediction_model' not in st.session_state:
        st.session_state['prediction_model'] = False
    if st.sidebar.button('Predict'):
        st.session_state['prediction_model'] = Pred_model
# Page navigation logic




 
target = data_description['target']
if st.session_state['prediction_model']:
    st.subheader(f"Predicting {target} using {st.session_state['prediction_model']}")
 
    model=load_pred_model(st.session_state['prediction_model']) 
   
 
     
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if data_description['Data_type'] == 'Classification':
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average=average_type))
        st.write("Recall:", recall_score(y_test, y_pred, average=average_type))
        st.write("F1 Score:", f1_score(y_test, y_pred, average=average_type))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report explanation:")
        st.button('Generate Metrics Explainer', key='metrics_explainer')
        if st.session_state.metrics_explainer:
            prompt = f"""
            Using {st.session_state["prediction_model"]} for {data_description} to predict {target}, 
            the following metrics were obtained: 
            - Accuracy: {accuracy_score(y_test, y_pred)} 
            - Precision: {precision_score(y_test, y_pred)} 
            - Recall: {recall_score(y_test, y_pred)} 
            - F1 Score: {f1_score(y_test, y_pred)}

            Explain these metrics, identify any that are below general standards, and suggest potential causes for low scores.
            """ 
            with st.spinner('Generating Metrics Explainer...'):
                Response = Gemini_model.generate_content(prompt).text           
                st.write_stream([x for x in Response])
                  
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title='Predicted vs. Actual')
        fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal'))
        st.plotly_chart(fig)
        residuals = y_test - y_pred

        # Scatter plot of residuals
        fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residual'}, title='Residual Plot')
        fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], mode='lines', name='Zero Residual'))
        st.plotly_chart(fig)

        st.subheader(f"Mean Squared Error  : {mean_squared_error(y_test, y_pred):.2f}")
        st.subheader(f"R2 Score       :  {r2_score(y_test, y_pred):.2f}")
 
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            
            
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            features = importance_df['Feature'][:10]
            # Use Plotly to create a bar plot for feature importance
            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title='Feature Importance for Random Forest',
                labels={'Feature': 'Features', 'Importance': 'Importance'},
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("Feature Importance:")
            st.write(features.tolist())
            features = features.tolist()
           
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if data_description['Data_type'] == 'Classification':
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
                st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
                st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
            else:
                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title='Predicted vs. Actual')
                fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal'))
                st.plotly_chart(fig)
                residuals = y_test - y_pred

                # Scatter plot of residuals
                fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residual'}, title='Residual Plot')
                fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], mode='lines', name='Zero Residual'))
                st.plotly_chart(fig)

                st.subheader(f"Mean Squared Error              : {mean_squared_error(y_test, y_pred):.2f}")
                st.subheader(f"R2 Score                   :  {r2_score(y_test, y_pred):.2f}")
        else:
            st.write("This model does not provide feature importances.")

        st.write('------ ')
        # Convert the DataFrame to CSV in memory
        df=pd.DataFrame(y_pred,columns=['Predicted'])
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Create a download button
        st.download_button(
            label='Download Predictions as CSV',
            data=csv_data,
            file_name='sample_data.csv',  # Default name for the downloaded file
            mime='text/csv'  # MIME type for CSV files
        )

        st.write("Download the CSV file by clicking the button above.")

        


        


