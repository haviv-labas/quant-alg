import streamlit as st
import pandas as pd
import altair as alt
import glob

# Title of the Streamlit app
st.title('NVDA Stock Open Price Visualization')




# List all CSV files in the ./data/ directory
csv_files = glob.glob('./data/*.csv')
csv_file_names = [file.split('/')[-1] for file in csv_files]  # Extract just the filenames for display

# If there are no CSV files found, display a message
if not csv_files:
    st.error('No CSV files found in the ./data/ directory.')

else:
    # Use a selectbox to list the CSV files
    selected_file_name = st.selectbox('Select a CSV file:', csv_file_names)
    # Get the full path of the selected file
    selected_file_path = './data/' + selected_file_name

# Load the selected CSV file into a pandas DataFrame
df = pd.read_csv(selected_file_path)

# Display the first few rows of the DataFrame to verify it's loaded correctly
print(df.head())

# Columns to check
columns_to_check = ['Date', 'Open', 'Close']

# Check if all specified columns exist in the DataFrame
are_all_columns_present = all(column in df.columns for column in columns_to_check)

# Convert "Date" from string to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Creating an Altair chart
chart = alt.Chart(df).mark_line().encode(
    x='Date:T',
    y='Open:Q',
    tooltip=['Date:T', 'Open:Q']
).properties(
    width=700,
    height=400,
    title='NVDA Open Price Over Time'
)

# Display the chart in the Streamlit app
st.altair_chart(chart, use_container_width=True)

