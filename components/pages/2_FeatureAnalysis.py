import os 
import plotly.express as px
import streamlit as st 
import pandas as pd
 

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.experimental import enable_iterative_imputer as IterativeImputer
from category_encoders import TargetEncoder
import numpy as np


from logger import logging
from utils import load_object,save_objects,initialize_artifact_folder,check_artifact_folder
from Home import  Gemini_model
# Path to the artifact folder


st.set_page_config(page_title="Feature Analysis", page_icon="🔍", layout='wide')



# Plot histogram for a feature
def plot_unique_values(df, feature, target):
   
    # Calculate mean target value for each unique value of the feature
    mean_target_values = df.groupby(feature)[target].mean().reset_index()

    # Create histogram plot with mean target values as text annotations
    fig = px.histogram(df, x=feature, title=f"Distribution of {feature}", color=feature)
    fig.update_traces(text=mean_target_values[target].round(2), textposition='outside')

    # Customize layout to display mean target values as annotations
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title='Count',
        showlegend=False,
        annotations=[
            dict(
                x=val,
                y=0,
                text=f"Mean Target: {mean:.2f}",
                showarrow=True,
                arrowhead=0,
                ax=0,
                ay=-40
            )
            for val, mean in zip(mean_target_values[feature], mean_target_values[target])
        ]
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig)

# Display missing values
def display_missing_values(df, feature):
    missing_percentage = (df[feature].isna().sum() / df.shape[0]) * 100  # Missing values as a percentage
    non_missing_percentage = 100 - missing_percentage
    data = {
    "Status": ["Missing", "Non-Missing"],
    "Percentage": [missing_percentage, non_missing_percentage],
}
    missing_df = pd.DataFrame(data)
    fig = px.bar(
    missing_df,
    x='Status',
    y='Percentage',
    title=f"Percentage of Missing vs. Non-Missing Values for {feature}",
    color='Status',
    color_discrete_map={"Missing": "red", "Non-Missing": "green"},
    
)
    st.plotly_chart(fig)

    
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return v

# Function to perform Chi-Square Test
def chi_square_test(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p





def impute_categorical_features(df, categorical_features, method='most_frequent', fill_value='Missing'):
    df_imputed = df.copy()

    # Initialize SimpleImputer with the specified method and fill_value
    if method == 'most_frequent':
        categorical_imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'constant':
        categorical_imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
    else:
        raise ValueError("Invalid imputation method. Use 'most_frequent' or 'constant'.")

    # Iterate over each categorical feature and impute missing values
    for feature in categorical_features:
        if df_imputed[feature].dtype == 'object':  # Check if the feature is categorical (object type)
            df_imputed[feature] = categorical_imputer.fit_transform(df_imputed[[feature]])

    return df_imputed
 
def impute_numerical_features(df, numerical_features, method='mean'):
    df_imputed = df.copy()

    # Initialize SimpleImputer with the specified method
    if method == 'mean':
        numerical_imputer = SimpleImputer(strategy='mean')
    elif method == 'most_frequent':
        numerical_imputer = SimpleImputer(strategy='most_frequent')
    else:
        raise ValueError("Invalid imputation method. Use 'mean' or 'most_frequent'.")

    # Iterate over each numerical feature and impute missing values
    for feature in numerical_features:
        if pd.api.types.is_numeric_dtype(df_imputed[feature].dtype):  # Check if the feature is numeric
            df_imputed[feature] = numerical_imputer.fit_transform(df_imputed[[feature]])

    return df_imputed
#Encoding categorical features
def target_encoded_imputation(df, categorical_features, target_feature, method='mean'):
    """
    Perform target-encoded imputation on categorical features using the target variable.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the features.
    categorical_features (list): List of names of categorical feature columns to impute.
    target_feature (str): Name of the target variable column.
    method (str): The imputation method. Options are 'mean', 'median', 'mode', or 'constant'.

    Returns:
    pandas.DataFrame: The DataFrame with imputed categorical features using target encoding.
    """
    df_encoded = df.copy()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_encoded[categorical_features], df_encoded[target_feature], test_size=0.2, random_state=42)

    # Use target encoding to impute missing values in categorical features based on target variable
    target_encoder = TargetEncoder()
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    # Impute missing values using the specified method
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    elif method == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value='Missing')
    else:
        raise ValueError("Invalid imputation method. Use 'mean', 'median', 'mode', or 'constant'.")

    df_encoded.loc[X_train_encoded.index, categorical_features] = imputer.fit_transform(X_train_encoded)
    df_encoded.loc[X_test_encoded.index, categorical_features] = imputer.transform(X_test_encoded)

    return df_encoded

#Numerical imputation
def knn_imputation(df, n_neighbors=5):
    """
    Perform K-Nearest Neighbors (KNN) imputation on numerical features.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numerical features.
    n_neighbors (int): The number of nearest neighbors to use for imputation.

    Returns:
    pandas.DataFrame: The DataFrame with imputed numerical features using KNN imputation.
    """
    df_imputed = df.copy()

    # Initialize KNNImputer with the specified number of neighbors
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Fit and transform the imputer on the dataset
    df_imputed[df.columns] = imputer.fit_transform(df)

    return df_imputed

#Numerical imputation
def iterative_imputation(df, method='linear'):
    """
    Perform iterative imputation (MICE) on numerical features using regression models.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numerical features.
    method (str): The regression model to use for imputation. Options are 'linear', 'random_forest', 'decision_tree', etc.

    Returns:
    pandas.DataFrame: The DataFrame with imputed numerical features using iterative imputation.
    """
    df_imputed = df.copy()

    # Initialize IterativeImputer with the specified regression model
    imputer = IterativeImputer(estimator=method, random_state=0)

    # Fit and transform the imputer on the dataset
    df_imputed[df.columns] = imputer.fit_transform(df)

    return df_imputed

def encode_cat_feature(df, feature, method):
     
    if method == 'one-hot-encode':
        feature_data = df[[feature]]
        # Apply OneHotEncoder
        encoder = OneHotEncoder(sparse=False, drop='first')  # drop first to avoid multicollinearity
        encoded_data = encoder.fit_transform(feature_data)
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names([feature]))
        # Return the encoded feature as a Series (first column of the encoded DataFrame)
        return encoded_df.iloc[:, 0]
    elif method == 'label-encode':
        # Label Encoding
        # Extract the feature column
        feature_data = df[feature]
        # Apply LabelEncoder
        encoder = LabelEncoder()
        encoded_data = encoder.fit_transform(feature_data)
        # Return the encoded feature as a Series
        return pd.Series(encoded_data, name=feature)

    else:
        raise ValueError("Unsupported encoding method. Choose either 'one-hot-encode' or 'label-encode'.")

def remove_outliers(df, numerical_features, threshold=1.5):
 
    df_cleaned = df.copy()

    # Iterate over each numerical feature and identify outliers using IQR method
    for feature in numerical_features:
        if pd.api.types.is_numeric_dtype(df_cleaned[feature].dtype):  # Check if the feature is numeric
            Q1 = df_cleaned[feature].quantile(0.25)
            Q3 = df_cleaned[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Filter out values that are outside the bounds defined by IQR method
            df_cleaned = df_cleaned[(df_cleaned[feature] >= lower_bound) & (df_cleaned[feature] <= upper_bound)]

    return df_cleaned

def scale_numerical_features(df, numerical_features, method='standardization'):
    
    df_scaled = df.copy()

    if method == 'standardization':
        scaler = StandardScaler()
    elif method == 'min-max':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Use 'standardization' or 'min-max'.")

    df_scaled[numerical_features] = scaler.fit_transform(df_scaled[numerical_features])

    return df_scaled

def apply_math_operations(df, numerical_features, operations):
    """
    Apply mathematical operations to numerical features of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numerical features.
    numerical_features (list): List of names of numerical feature columns to apply operations on.
    operations (dict): Dictionary specifying the operations to apply for each feature.
                      Example: {'feature1': ('log',), 'feature2': ('exp',), 'feature3': ('+', 'feature1', 'feature2')}

    Returns:
    pandas.DataFrame: The DataFrame with modified numerical features after applying operations.
    """
    df_modified = df.copy()

    for feature, op_args in operations.items():
        if feature in numerical_features:
            operation = op_args[0]  # Get the operation (e.g., '+', '*', 'log', 'exp')
            if operation == '+':
                # Addition operation
                other_feature = op_args[1]
                df_modified[feature] = df_modified[feature] + df_modified[other_feature]
            elif operation == '-':
                # Subtraction operation 
                other_feature = op_args[1]    
                df_modified[feature] = df_modified[feature] - df_modified[other_feature]
            elif operation == '*':
                # Multiplication operation
                other_feature = op_args[1]
                df_modified[feature] = df_modified[feature] * df_modified[other_feature]
            elif operation == '/':
                # Division operation
                other_feature = op_args[1]
                df_modified[feature] = df_modified[feature] / df_modified[other_feature]

            elif operation == 'log':
                # Logarithm operation (natural logarithm)
                df_modified[feature] = np.log(df_modified[feature])
            elif operation == 'exp':
                # Exponential operation (e^x)
                df_modified[feature] = np.exp(df_modified[feature])
            elif operation == 'sqrt':
                # Square root operation
                df_modified[feature] = np.sqrt(df_modified[feature])

            else:
                raise ValueError(f"Unsupported operation: {operation}")

    return df_modified



def Encode_data(data,feature,Encode_option):
    if Encode_option == 'One-HotEncode':
        modified_data = data.copy()
        modified_data = pd.get_dummies(modified_data, columns=[feature], drop_first=True)
        return modified_data[feature]
    elif Encode_option == 'LabelEncode':
        modified_data = data.copy()
        modified_data[feature] = modified_data[feature].astype('category').cat.codes
        return modified_data[feature]

def impute_data(data,feature):
    if data[feature].dtype == 'object':
        modified_data = data.copy()
        modified_data[feature] = modified_data[feature].fillna('missing')
        return modified_data[feature]
    else:
        modified_data = data.copy()
        modified_data[feature] = modified_data[feature].fillna(modified_data[feature].mean())
        return modified_data[feature]

def check_untracked_features(df):
    cat_features = []
    features_with_missing_values = []

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Check if the column data type is categorical (object or categorical)
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            cat_features.append(column)
        
        # Check if the column has any missing values
        if df[column].isnull().sum() > 0:
            features_with_missing_values.append(column)
    
    # Create a dictionary to store the results
    result_dict = {
        'unimputed_cat_features': cat_features,
        'features_with_missing_values': features_with_missing_values
    }
    
    if len(cat_features) == 0 and len(features_with_missing_values) == 0:
        return None        
    else:
        return result_dict
    
def remove_duplicate_transformations_from_dict(transformations_dict):
    filtered_dict = {}
    if transformations_dict is not None:
        for feature, transformations in transformations_dict.items():
            seen_transformations = set()
            unique_transformations = []

            for transformation in transformations:
                # Extract the part of the transformation before the first underscore
                # Split the transformation string by underscore ('_')
                parts = transformation.split('_')
                
                # Take the first part of the split (before the first underscore)
                if parts:
                    base_transformation = parts[0]
                else:
                    base_transformation = transformation  # Use full transformation if no underscore is found

                # Check if the base transformation has already been seen
                if base_transformation not in seen_transformations:
                    # Add the base transformation to the set of seen transformations
                    seen_transformations.add(base_transformation)
                    # Add the full transformation to the list of unique transformations
                    unique_transformations.append(transformation)

            # Store the filtered transformations in the filtered_dict
            filtered_dict[feature] = unique_transformations

        return filtered_dict
    else:
        return None

def extract_features_to_remain(unimputed_cat_features, action_log):
    features_to_remain = []

    # Iterate over each feature in unimputed_cat_features
    for feature in unimputed_cat_features:
        # Check if the feature is not in action_log
        if feature not in action_log:
            features_to_remain.append(feature)

    return features_to_remain

def split_action(action):
    # Split the action string at the underscore ('_')
    parts = action.split('_')
    
    if len(parts) == 2:
        method_type, method_name = parts
        return method_type, method_name
    else:
        # Handle unexpected format (e.g., if the action string does not contain exactly one underscore)
        return None, None
def sort_dict_by_custom_order(input_dict, custom_order):
    # Filter custom order list to include only keys present in the input dictionary
    valid_custom_order = [key for key in custom_order if key in input_dict]
    
    # Create a new dictionary with keys sorted according to the filtered custom order
    sorted_dict = {key: input_dict[key] for key in valid_custom_order}
    
    return sorted_dict

def impute_and_encode(df):
    # Impute missing values in numerical features with mean
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    # Encode categorical features with label encoding
    categorical_features = df.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        df[feature] = label_encoder.fit_transform(df[feature])

    return df    

def commit_changes(df, action_log, name, target, description):
    custom_order = [
    "Drop",
    "CatImpute_Most frequent",
    "CatImpute_constant",
    "CatImpute_Target encoded",
    "CatEncode_One-HotEncode",
    "CatEncode_LabelEncode",
    "NumScale_MinMax",
    "NumImpute_Mean",
    "NumImpute_Median",
    "NumImpute_KNN",
    "NumImpute_Iterative",
    "RemoveOutliers"
]


    list_of_methods = {}
    for feature in action_log:
        index = list(feature.keys())[0]
        st.write(index, feature[index])
        if feature[index] not in list_of_methods:
            list_of_methods[feature[index]] = []
        if index not in list_of_methods[feature[index]]:
            list_of_methods[feature[index]].append(index)
    

    sorted_list_of_methods = sort_dict_by_custom_order(list_of_methods, custom_order)
    

    for method, features in sorted_list_of_methods.items():
        try:
            if method == 'Drop':
                df.drop(features, axis=1, inplace=True)
            elif method == 'RemoveOutliers':
                df = remove_outliers(df, features, threshold=1.5)
            elif method == 'NumScale_Min-Max':
                df = scale_numerical_features(df, features, method='min-max')
            elif method == 'CatImpute_Most frequent':
                df = impute_categorical_features(df, features, method='most_frequent')
            elif method == 'CatImpute_constant: Missing':
                df = impute_categorical_features(df, features, method='constant', constant_value='Missing')
            elif method == 'CatImpute_Target encoded imputation':
                df = target_encoded_imputation(df, features)
            elif method == 'CatEncode_One-HotEncode':
                df = encode_cat_feature(df, features, method='one-hot-encode')
            elif method == 'CatEncode_LabelEncode':
                df = encode_cat_feature(df, features, method='label-encode')
            elif method == 'NumImpute_Mean':
                df = impute_numerical_features(df, method='mean')
            elif method == 'NumImpute_Median':
                df = impute_numerical_features(df, method='most_frequent')
            elif method == 'NumImpute_KNN':
                df = knn_imputation(df, n_neighbors=5)
            elif method == 'NumImpute_Iterative':
                df = iterative_imputation(df, method='linear')
            else:
                # Handle unrecognized method (if needed)
                pass
        except Exception as e:
            df = impute_and_encode(df)
            logging.error(f"Error generating New Data: {e}")
    data_path = os.path.join("artifact", f"Modified_{name}.csv")
    description = {
                "name":name,
                "target": target,
                "Data_type":description
                }  
    description_path = os.path.join("artifact", f"Modified_{name}_description.pkl")
    save_objects(description_path,description)   
    save_objects(data_path,df) 
    col1,col2 = st.columns([0.5,1.5])
    with col1:
        st.info('New Data Saved Succesfully ! ')       


def plot_feature_info(data,feature,target,name,description):
     
    if 'log' not in st.session_state or st.sidebar.button('Reset Log') :
        st.session_state.log = []
    modified_data = data.copy()
    modified_data_path= os.path.join("artifact", f"Modified_{name}.csv")
    feature_type = data[feature].dtype
    st.subheader(f"Feature Type : {feature_type}")
    if feature_type == 'object':
        display_missing_values(data, feature)
        imputation_methode = st.selectbox("Select",['Most frequent','constant: Missing','Target encoded imputation'])
        if st.button('Impute Cat Feature'):
            st.write('Operation Logged')
            st.session_state.log.append({feature: f'CatImpute_{imputation_methode}'})
        st.write('----------')
        plot_unique_values(data, feature,target)
        Encode_option = st.selectbox("Select",['One-HotEncode','LabelEncode'])
        if st.button('Encode'):
            st.session_state.log.append({feature :f'CatEncode_{Encode_option}'})
            st.write('Operation Logged')

        st.button('Generate Encoding Suggestion')
        st.write('----------')
        if "cramer_explanation" not in st.session_state:
            st.session_state["cramer_explanation"] = None

        if "chi2_explanation" not in st.session_state:
            st.session_state["chi2_explanation"] = None

        # Calculate Cramér's V
        v = cramers_v(data[feature], data[target])
        st.write(f"Cramér's V between {feature} and {target}: {v:.2f}")
        if 'cramer_explanation' not in st.session_state:
            st.session_state.cramer_explanation= None

        if st.button("Generate cramer's V test Explination"):
            with st.spinner('Generating...'):
                cramer_explanation_text = Gemini_model.generate_content(f'in the context if {name} Dataset with brief description {description} explain the Cramér s V value between {feature} and {target} that is equal {v:.2f} ')
                st.session_state["cramer_explanation"] = cramer_explanation_text.text
                st.subheader("Cramer's V test explanation :")
                st.write_stream([x for x in st.session_state["cramer_explanation"]])    

        # Calculate Spearman's correlation
        # Perform and display Chi-Square Test results
        chi2, p = chi_square_test(data[feature], data[target])
        st.write(f"Chi-Square statistic: {chi2:.2f}")
        st.write(f"P-value: {p:.2e}")
       
        if "chi2_explanation"not in st.session_state:
            st.session_state["chi2_explanation"] = None
        if st.button('Generate chi2 test Explination'):
            with st.spinner('Generating...'):
                chi2_corr_text = Gemini_model.generate_content(f"in the context if {name} Dataset with brief description {description} explain Chi-Square test result between {feature} and {target} that is equal chi2{chi2:.2f} and p-value {p:.2e}")
                st.session_state["chi2_explanation"] = chi2_corr_text.text
                st.subheader("Chi-Square statistic's Explanation :")
                st.write_stream([x for x in st.session_state["chi2_explanation"]])    

    
    if feature_type in ['int64', 'float64']:
        
        display_missing_values(data, feature)
        imputation_methode = st.selectbox("Select",['Mean','Median','KNN','Iterative'])
         
        if st.button('Impute Num Feature'):
            st.write('Operation Logged')
            st.session_state.log.append({feature: f'NumImpute_{imputation_methode}'})
        st.write('----------')
        
        plot_unique_values(data, feature,target)
        #scatterplot
        fig = px.scatter(data, x=feature, y=target, title=f"Scatter plot of {feature} vs {target}")
        st.plotly_chart(fig)
        
        # Histogram
        fig = px.histogram(data, x=feature, nbins=30, title=f"Histogram of {feature}")
        st.plotly_chart(fig)
        if st.button('Min-Max Scale'):
            st.write('Operation Logged')
            st.session_state.log.append({feature: f'NumScale_Min-Max'})
         
        st.write('----------')

        # Box plot
        fig = px.box(data, y=feature, title=f"Box Plot of {feature}")
        st.plotly_chart(fig)
        if st.button('Remove Outliers'):
            st.write('Operation Logged')
            st.session_state.log.append({feature: f'RemoveOutliers'})
        st.write('----------')
        # Calculate Pearson's correlation
        pearson_corr = data[feature].corr(data[target], method='pearson')
        st.subheader(f"Pearson's correlation between {feature} and {target}: {pearson_corr:.2f}")
        if "pearson_explanation" not in st.session_state:
            st.session_state["pearson_explanation"] = None

        if "spearman_explanation" not in st.session_state:
            st.session_state["spearman_explanation"] = None

        if st.button("Generate Pearson's test Explination"):
            with st.spinner('Generating...'):
                pearson_corr_text = Gemini_model.generate_content(f"in the context if {name} Dataset with brief description {description} explain the Pearson correlation value between {feature} and {target} that is equal {pearson_corr:.2f} ")
                st.session_state["pearson_explanation"] = pearson_corr_text.text
                st.subheader("Pearson's Explanation :")
                st.write(st.session_state["pearson_explanation"])
        # Calculate Spearman's correlation
        spearman_corr = data[feature].corr(data[target], method='spearman')         
        st.subheader(f"Spearman's correlation between {feature} and {target}: {spearman_corr:.2f}")
        if st.button('Generate spearman test Explination'):
            with st.spinner('Generating...'):
                spearman_corr_text = Gemini_model.generate_content(f"in the context if {name} Dataset with brief description {description} explain theSpearman's correlation value between {feature} and {target} that is equal {spearman_corr:.2f} ")
                st.session_state["spearman_explanation"] = spearman_corr_text.text
                st.subheader("Spearman's Explanation :")
                st.write(st.session_state["spearman_explanation"])  

            
    st.write('----------------------')
    _,col2 = st.columns([1,1.65])
    with col2:
        if st.button('🗑️ Drop Feature ❌'):
            st.write('Operation Logged')
            st.session_state.log.append({feature: 'Drop'})          
    
    if st.session_state.log == {}:
        st.write('No Operation Logged')    
    
    if st.session_state.log:
        st.write('----------------------')
        st.subheader('Action Log:')
        feature_dict = {}
        for item in st.session_state.log:
            for key, value in item.items():
                if key in feature_dict:
                    if value not in feature_dict[key]:
                        feature_dict[key].append(value)
                else:
                    feature_dict[key] = [value]
        filterd_feature_dict = remove_duplicate_transformations_from_dict(feature_dict)            
        st.write(filterd_feature_dict)            
        untracked_feats = [x for x in data.columns if x not in feature_dict]
        if check_untracked_features(data)  is not None:
            if check_untracked_features(data)['unimputed_cat_features'] is not None and check_untracked_features(data)['features_with_missing_values'] is not None:
                untracked_feats = extract_features_to_remain(check_untracked_features(data)['unimputed_cat_features']+check_untracked_features(data)['features_with_missing_values'], filterd_feature_dict)
            
            if check_untracked_features(data)['unimputed_cat_features'] is  None and check_untracked_features(data)['features_with_missing_values'] is not None:
                untracked_feats = extract_features_to_remain(check_untracked_features(data)['features_with_missing_values'], filterd_feature_dict)
            
            if check_untracked_features(data)['unimputed_cat_features'] is not None and check_untracked_features(data)['features_with_missing_values'] is None:
                untracked_feats = extract_features_to_remain(check_untracked_features(data)['unimputed_cat_features'], filterd_feature_dict)           
           
        
      
        if untracked_feats is None:
            st.write('All features have been tracked')
        else:
            st.write(f'these features remain to be done :') 
            st.write(untracked_feats) 

        st.write('----------------------')
        _,col2 = st.columns([1,1.65])
        if st.session_state.log and untracked_feats==[]:
            with col2:
                if st.button('Commit Changes 📝'):
                    commit_changes(data, st.session_state.log, name,target,description)
                   



    
def select_file(artifact_folder='artifact'):

    initialize_artifact_folder(artifact_folder)
    artifact_files = check_artifact_folder(artifact_folder)
    selection_list = [x for x in artifact_files if x.endswith('.csv')]
    selected_data = st.sidebar.selectbox('Pick DataSet',selection_list)
    selected_data_description = selected_data.split('.')[0] + '_description.pkl'
    if artifact_files:
        # Load and display the first file based on its extension
        data_description = load_object(os.path.join(artifact_folder,selected_data_description))
        data = load_object(os.path.join(artifact_folder, selected_data))
        st.write("Data Description:", data_description)
        st.sidebar.subheader(f"Type : {data_description['Data_type']}")
    return data, data_description 

def main():
    
    data, data_description = select_file()

    # Page navigation logic
    st.title("Data Analysis App")
    target = data_description['target']

    feature_type = st.sidebar.selectbox("Pick a feature type", ['Categorical','Numerical'])
    if feature_type == 'Categorical':
        features_type = ['object']
    else:
        features_type = ['int64', 'float64']
    selection_feature = [x for x in data.columns if data[x].dtype in features_type and x != target] 
    selection_feature = selection_feature[::-1]
    feature = st.sidebar.selectbox("Pick a feature", selection_feature)
    
    st.write('')
    st.subheader('Ai generated Feature Description:')
    st.write('')
    if st.button('Generate Feature Description'):
        try:
            with st.spinner('Generating...'):
                feature_description = Gemini_model.generate_content(f"Describe the feature {feature} in the context of {data_description['name']} dataset with brief description {data_description['Data_type']}")
                st.write_stream([x for x in feature_description.text])
        except Exception as e:
            st.error(f"An error occurred: {e}")        
    st.write('----------------------')    
    plot_feature_info(data,feature,target,name=data_description['name'],description=data_description['Data_type'])
 
     
        
  

if __name__ == "__main__":
    main()