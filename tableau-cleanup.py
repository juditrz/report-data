import streamlit as st
import pandas as pd
import io

# Set up the Streamlit app
st.title("Monthly Report Data Organizer")
st.write("Upload your monthly report Tableau files for cleanup and merging.")

# File upload
organic_file = st.file_uploader("Upload organic data Excel file", type=["xlsx"])
paid_file = st.file_uploader("Upload paid data Excel file", type=["xlsx"])

# User options
remove_backslash = st.checkbox("Remove trailing backslash from URLs", value=True)
replace_amp = st.checkbox("Remove AMP from URLs", value=True)

if organic_file is not None and paid_file is not None:
    try:
        # Read Excel files
        df1 = pd.read_excel(organic_file, header=None)
        df2 = pd.read_excel(paid_file, header=None)

        # Define columns to drop
        columns_to_drop = [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15]

        # Drop the specified columns
        df1.drop(df1.columns[columns_to_drop], axis=1, inplace=True)
        df2.drop(df2.columns[columns_to_drop], axis=1, inplace=True)

        # Assume the first column (index 0) is 'Post URL' in both DataFrames
        df1.columns = ['Post URL'] + [f'Organic_{i}' for i in range(1, df1.shape[1])]
        df2.columns = ['Post URL'] + [f'Paid_{i}' for i in range(1, df2.shape[1])]

        # Merge the data based on the 'Post URL' column using an outer join
        merged_df = pd.merge(df1, df2, on='Post URL', how='outer')

        # Replace NaN values with 0 for all relevant columns
        organic_columns = [col for col in merged_df.columns if 'Organic' in col]
        paid_columns = [col for col in merged_df.columns if 'Paid' in col]

        merged_df[organic_columns] = merged_df[organic_columns].fillna(0)
        merged_df[paid_columns] = merged_df[paid_columns].fillna(0)

        # Apply user-selected functions
        merged_df['Post URL'] = merged_df['Post URL'].replace('https://wi', 'https://www.wi', regex=True)
        if replace_amp:
            merged_df['Post URL'] = merged_df['Post URL'].replace('/amp/', '/', regex=True)
        if remove_backslash:
            merged_df['Post URL'] = merged_df['Post URL'].apply(lambda x: x[:-1] if isinstance(x, str) else x)

        # Create a pivot table
        pivot_table = merged_df.pivot_table(
            index='Post URL',  # Rows
            values=[col for col in merged_df.columns if col != 'Post URL'],  # Values to aggregate
            aggfunc='sum',  # Aggregation function
            fill_value=0  # Fill missing values with 0
        )

        # Convert pivot table to DataFrame and reset index
        result_df = pivot_table.reset_index()

        # Interleave the columns and calculate the total columns
        categories = ['Anon', 'Free', 'Premium', 'Collections']
        for i, category in enumerate(categories):
            organic_col = f'Organic_{i+1}'
            paid_col = f'Paid_{i+1}'
            total_col = f'{category} total'

            result_df[f'{category} organic'] = pd.to_numeric(result_df[organic_col], errors='coerce').fillna(0)
            result_df[f'{category} paid'] = pd.to_numeric(result_df[paid_col], errors='coerce').fillna(0)
            result_df[total_col] = result_df[f'{category} organic'] + result_df[f'{category} paid']

        # Add new columns for Free/Anon and Prem. rate
        result_df['Free/Anon organic'] = result_df['Free organic'] / result_df['Anon organic'].replace(0, pd.NA)
        result_df['Free/Anon paid'] = result_df['Free paid'] / result_df['Anon paid'].replace(0, pd.NA)
        result_df['Free/Anon avg.'] = result_df['Free total'] / result_df['Anon total'].replace(0, pd.NA)

        result_df['Prem. rate organic'] = result_df['Premium organic'] / result_df['Free organic'].replace(0, pd.NA)
        result_df['Prem. rate paid'] = result_df['Premium paid'] / result_df['Free paid'].replace(0, pd.NA)
        result_df['Prem. rate avg.'] = result_df['Premium total'] / result_df['Free total'].replace(0, pd.NA)

        # Add new calculated columns to the processed DataFrame
        result_df['SHARK'] = result_df['Anon total'] * result_df['Free total'] * result_df['Premium total']
        result_df['$hark'] = result_df['Free total'] * result_df['Premium total']
        result_df['oShark'] = result_df['Anon organic'] * result_df['Free organic'] * result_df['Premium organic']
        result_df['pShark'] = result_df['Anon paid'] * result_df['Free paid'] * result_df['Premium paid']

        # Replace the top 50 values in SHARK, $hark, oShark, and pShark with the column name and set others to empty
        columns_to_process = ['SHARK', '$hark', 'oShark', 'pShark']
        for column in columns_to_process:
            # Get the top 50 indices
            top_indices = result_df[column].nlargest(50).index
            # Convert the column to object type to allow mixed types
            result_df[column] = result_df[column].astype(object)
            # Replace top 50 values with the column name
            result_df.loc[top_indices, column] = column
            # Set other values to empty
            result_df.loc[~result_df.index.isin(top_indices), column] = ''

        # Define the order of the columns as specified
        ordered_columns = [
            'Post URL',
            'Anon organic', 'Anon paid', 'Anon total',
            'Free organic', 'Free paid', 'Free total',
            'Free/Anon organic', 'Free/Anon paid', 'Free/Anon avg.',
            'Premium organic', 'Premium paid', 'Premium total',
            'Prem. rate organic', 'Prem. rate paid', 'Prem. rate avg.',
            'Collections organic', 'Collections paid', 'Collections total',
            'SHARK', '$hark', 'oShark', 'pShark'
        ]

        # Reorder the columns
        result_df = result_df[ordered_columns]

        # Sort the DataFrame by 'Anon total' in descending order
        result_df = result_df.sort_values(by='Anon total', ascending=False)

        # Calculate and display the sum of specific columns
        columns_to_sum = ['Anon total', 'Free total', 'Premium total', 'Collections total']
        sums = result_df[columns_to_sum].sum()

        # Rename the index
        sums.index = ['Anons', 'Frees', 'Premiums', 'Collections']

        # Format the columns
        sums_formatted = sums.apply(lambda x: f"${int(x):,}" if 'Collections' in sums.index[sums == x] else f"{int(x):,}")

        st.write("### Monthly data overview")
        sum_df = pd.DataFrame({
            "Metric": sums.index,
            "Total": sums_formatted.values
        })
        st.write(sum_df)

        # Provide a download link for the processed data
        st.write("### Download processed data file")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False)
            writer.close()
        st.download_button(
            label="Download as Excel",
            data=buffer.getvalue(),
            file_name="monthly_tableau_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
