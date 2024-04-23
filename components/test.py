import streamlit as st
import re  # For regular expression-based validation

# Function to validate and format file names
def validate_file_name(raw_name):
    # Disallowed characters that should not be in a file name
    forbidden_chars = r'[\ / ? % * : | " < >]'
    
    # Replace spaces with underscores and remove forbidden characters
    formatted_name = re.sub(forbidden_chars, "", raw_name.strip().replace(" ", "_"))

    return formatted_name


st.title("File Name Input Validator")

# Get user input for file name
file_name_input = st.text_input("Enter a raw file name:", "my file 2024.doc")

# Validate and format the input file name
formatted_file_name = validate_file_name(file_name_input)

# Check if the formatted file name matches the original input
if file_name_input.strip().replace(" ", "_") == formatted_file_name:
    st.success(f"Valid file name: '{formatted_file_name}'")
else:
    st.warning(
        f"The file name '{file_name_input}' contained invalid characters or extra spaces. It has been formatted to '{formatted_file_name}'."
    )

# Display the formatted file name
st.write("Formatted File Name:")
st.code(formatted_file_name)
