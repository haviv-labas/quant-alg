import pandas as pd
import altair as alt

# Load the CSV file into a DataFrame
df = pd.read_csv('data/NVDA.csv')

# Display the first few rows of the DataFrame to verify it's loaded correctly
print(df.head())

# Columns to check
columns_to_check = ['Open', 'Close']

# Check if all specified columns exist in the DataFrame
are_all_columns_present = all(column in df.columns for column in columns_to_check)


print(df.describe())
print(df["Date"])
print(f"Are all specified columns present? {are_all_columns_present}")



# Convert "Date" from string to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plotting with Altair
chart = alt.Chart(df).mark_line().encode(
    x='Date:T',  # T indicates temporal type for better axis formatting
    y='Open:Q',  # Q indicates a quantitative scale
    tooltip=['Date:T', 'Open:Q']  # Tooltip for detailed info on hover
).properties(
    width=600,
    height=300,
    title='NVDA Open Price Over Time'
)

chart.display()
