import streamlit as st

# Define the about page
def about_page():
    st.title("About Our Data Analysis App 🌟")
    st.write("""
    Welcome to our Data Analysis App! This app is designed to help you analyze data, create predictions, and interact with a smart assistant for AI-driven insights. Let's dive into the key features of our app, which is comprised of four main pages: **Home**, **Feature Analysis**, **Prediction**, and **Assistant**.

    ## Home 🏛️
    The Home page is where you set the foundation for your analysis:
    - Enter your **Google API key** 🔑 to unlock additional features.
    - Import up to **5 CSV datasets** 📂.
    - Specify the **data type** (regression or classification) and add a brief description. This helps the LLM (Large Language Model) offer more tailored suggestions throughout the app.
    - Choose your **target variable** 🎯 and hit **Load** to initialize the data. The app will provide a data description along with information about your target feature and other dataset details.

    ## Feature Analysis 📊
    On this page, you can delve into the details of your data:
    - Choose the dataset and specify each feature's **type** (categorical or numerical).
    - Select the features you want to analyze and generate their description, including **imputation** and **encoding** methods.
    - Get tailored suggestions from **Google's Gemini Pro 1.5 LLM** 🤖 to improve your data preparation.
    - Once the dataset is fully encoded and non-empty, you're ready for the next step!

    ## Prediction 🔍
    In this section, you can create predictions and analyze model performance:
    - Pick the dataset you want to use for predictions.
    - The app automatically selects a model based on the data type (regression or classification).
    - Click **Predict** to generate predictions, and the app will present general metrics about the chosen model.
    - These metrics are further explained with help from Gemini Pro 1.5 LLM, ensuring you understand the results and can make informed decisions.

    ## Assistant 💬
    The Assistant page is like your personal AI expert:
    - A chatbot powered by **Google's Gemini Pro 1.5** allows you to ask questions about data analysis, predictions, and more.
    - You can customize the chatbot's behavior by setting **temperature**, **max length**, and **top p** parameters.
    - Get answers to your questions and suggestions for improving your data analysis process.

    ## Technologies Used 💻
    This app leverages several technologies to deliver a seamless user experience:
    - **Streamlit** for building an interactive web application.
    - **Google's Gemini Pro 1.5** for AI-driven insights and chatbot functionality.
    - **Python** as the main programming language for data analysis and backend processing.
    - **Pandas** and **NumPy** for data manipulation and analysis.
    - **Scikit-Learn** for building machine learning models.

    We hope you enjoy using our Data Analysis App and find it valuable for your projects. If you have any questions or suggestions, feel free to reach out through the Assistant page. Happy analyzing! 📊
    """)

# To use this in Streamlit, call about_page() within a Streamlit app
about_page()
