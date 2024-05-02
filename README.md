# Data Analysis and Prediction App with LLM-Based Assistant 🚀
This application provides comprehensive tools for data analysis and feature engineering, along with predictive modeling capabilities. Additionally, it includes an assistant chatbot that uses Llama 3 and Google Gemini to offer insightful guidance during your data science journey.
## See Demo 🌐 [Click here](https://data-analysis-app-geminpro.streamlit.app)
![Project Picture](https://github.com/BelhsanHmida/Data-Analysis-and-Prediction-App-with-LLM-Based-Assistant/blob/main/Capture.PNG)
## Table of Contents 📚
1. [Features](#features) 🌟
2. [Getting Started](#getting-started) 🏁
3. [Installation](#installation) 🔧
4. [Usage](#usage) 💻
5. [Contributing](#contributing) ✨
6. [License](#license) 📜
7. [Contact](#contact) 📬

## Features ✨
- Import and manage datasets in CSV format.
- Set data types (regression or classification) and provide dataset descriptions.
- Conduct feature analysis, including imputation and encoding and get llm based suggestions .
- Generate predictive models and view key metrics and their explanation .
- Interact with a chatbot powered by Google's Gemini Pro 1.5 for AI-driven Help.
  
## Libraries and Technologies Used 🛠️
- [**Streamlit**](https://streamlit.io/): For building an interactive web application.
- [**Google's Gemini Pro 1.5**](https://blog.google/products/ai/gemini-ai): The underlying large language model for AI-based insights and the chatbot.
- [**Plotly**](https://plotly.com/): For interactive data visualizations.
- [**Scikit-Learn**](https://scikit-learn.org/stable/): For building machine learning models.
- [**Scipy**](https://www.scipy.org/): For scientific computing and additional data manipulation tools.
- [**Pandas**](https://pandas.pydata.org/): For data manipulation and analysis.
- [**NumPy**](https://numpy.org/): For numerical computing.

## End-to-End Design 🏁
This app is designed with an end-to-end approach, ensuring smooth transitions between different stages of data analysis:
- **Home**: Set up Google API keys, import datasets, and initialize analysis.
- **Feature Analysis**: Define feature types and get suggestions for imputation and encoding.
- **Prediction**: Build predictive models based on the data type and view key metrics.
- **Assistant**: Interact with a smart chatbot for additional insights and guidance.

## Object-Oriented Design 🔄
The code is organized using object-oriented principles, ensuring maintainability and scalability:

Modular Structure: Each feature and function is encapsulated within its module, promoting a clean and organized codebase.
Code Reusability: Utility functions are housed in a separate module, allowing for code reuse across the app.
Exception Handling: Custom exceptions and detailed logging ensure that errors are caught and addressed efficiently.

# How to Use the App Localy 📝 

1. Clone this Repository on your local machine
2. Create a virtual environment `conda create -n venv python=3.9` 
3. Activate it `conda activate venv`
4. Install initial deps `pip install requirements.txt`
5. Run the app `🏛️_Home.py`

# Credits :
  This project was developed by Mohamed Hmida.
# License:
  This project is licensed under the Apache license 2.0  License.
   
