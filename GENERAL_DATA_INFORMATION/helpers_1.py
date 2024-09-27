from dash import dcc, html, dash_table as dt
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from helpers import get_metadata

import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Function to calculate general data summary and create the Bootstrap table layout
def general_data_summary_table(data_id):
    # Get the data
    metadata, data, name = get_metadata(data_id)
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
    
    
    
    # Need to convert the data to a dataframe we can work with
    dataframe = data.get_data()[0]
    
    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    n_rows = dataframe.shape[0]
    n_features = dataframe.shape[1]
    
    data_type_counts = metadata.groupby('DataType').size().reset_index(name='Value')

    # Add a new column 'Statistic' with the description
    data_type_counts['Statistic'] = 'Number of ' + data_type_counts['DataType'].astype(str) + ' features'

    # Reorder columns to have 'Statistic' first
    data_type_table = data_type_counts[['Statistic', 'Value']]
        
    stats = {
        "Statistic": ["Number of rows", "Number of features"],
        "Value": [n_rows, n_features]
    }
    
    # Creating a dataframe for "Number of Rows" and "Number of Features"
    summary_df = pd.DataFrame({
        'Statistic': ['Number of rows', 'Number of features'],
        'Value': [n_rows, n_features]
    })

    # Concatenating the two dataframes
    df_stats = pd.concat([summary_df, data_type_table], ignore_index=True)
    
    table_header = [html.Thead(html.Tr([html.Th(col) for col in df_stats.columns]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(df_stats.iloc[i][col]) for col in df_stats.columns])
        for i in range(len(df_stats))
    ])]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)



# Function to calculate data quality summary and create the Bootstrap table layout
def data_quality_summary_table(data_id):
    # Get the data
    metadata, data, name = get_metadata(data_id)
    
    # Need to convert the data to a dataframe we can work with
    dataframe = data.get_data()[0]
    
    num_missing_values = dataframe.isnull().sum().sum()
    num_columns_with_missing_values = dataframe.isnull().sum().gt(0).sum()
    num_duplicate_rows = dataframe.duplicated().sum()
    num_duplicate_columns = dataframe.columns.duplicated().sum()
    
    stats = {
        "Statistic": [
            "Number of missing values", 
            "Number of columns with missing values", 
            "Number of duplicate rows", 
            "Number of duplicate columns"
        ],
        "Value": [
            num_missing_values, 
            num_columns_with_missing_values, 
            num_duplicate_rows, 
            num_duplicate_columns
        ]
    }
    
    df_stats = pd.DataFrame(stats)
    
    table_header = [html.Thead(html.Tr([html.Th(col) for col in df_stats.columns]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(df_stats.iloc[i][col]) for col in df_stats.columns])
        for i in range(len(df_stats))
    ])]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)

import pandas as pd
import numpy as np

# Function to create the metadata table for the nominal features in the dataset
def create_nominals_metadata_table(data_id):
    # Get the data
    metadata, data, name = get_metadata(data_id)
    
    # Need to convert the data to a dataframe we can work with
    dataframe = data.get_data()[0]
    
    # Filter nominal data types
    metadata_nominal = metadata[metadata.DataType == "nominal"]
    
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    
    # Create summary table with mode values
    summary_table_nominal = pd.DataFrame({
        'Attribute': metadata_nominal.Attribute.astype(str),  # Ensure Attribute is a string
        'Mode': [dataframe[col].mode()[0] for col in metadata_nominal.Attribute],
        # Calculate entropy for each nominal feature and round to 2 decimal places
        'Entropy': [
            round(
                -np.sum((dataframe[col].value_counts(normalize=True) * 
                         np.log2(dataframe[col].value_counts(normalize=True) + 1e-9))
                ), 2
            )
            for col in metadata_nominal.Attribute
        ]
    })
    
    # Merge metadata_nominal with summary_table_nominal on 'Attribute'
    metadata_nominal_renewed = pd.merge(
        metadata_nominal[["Attribute", "DataType", "# categories", "Missing values", "Target"]].astype({'Attribute': 'str'}),
        summary_table_nominal,
        on='Attribute',
        how='inner'
    )
    # Check if the target_feature is present and rename it accordingly
    metadata_nominal_renewed['Attribute'] = metadata_nominal_renewed['Attribute'].apply(
        lambda x: f"{x}(target)" if x == target_feature else x
    )
    
    return metadata_nominal_renewed[["Attribute", "# categories", "Mode", "Missing values", "Entropy"]]


# Function to create the metadata table for the numeric features in the dataset
def create_numerics_metadata_table(data_id):
    # Get the data
    metadata, data, name = get_metadata(data_id)
    
    # Need to convert the data to a dataframe we can work with
    dataframe = data.get_data()[0]
    
    # Filter numeric data types
    metadata_numeric = metadata[metadata.DataType == "numeric"]
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    
    # Check if there are any numeric features
    if metadata_numeric.empty:
        print("No numeric features found in metadata.")
        return pd.DataFrame()  # Return an empty DataFrame if no numeric features
    
    # Filter dataframe columns that are numeric and present in metadata
    numeric_features = list(metadata_numeric.Attribute)
    
    # Check if there are numeric features in the dataframe
    numeric_data = dataframe[numeric_features]
    if numeric_data.empty:
        print("No numeric data found in the dataframe.")
        return pd.DataFrame()  # Return an empty DataFrame if no numeric data
    
    # Create summary table with min, max, mean, median, std values, checking that numeric data exists
    summary_table_numerics = numeric_data.agg(['min', 'max', 'mean', 'median', 'std']).transpose()
    
    # Add a column named "Attribute" with the names of the numeric features
    summary_table_numerics['Attribute'] = summary_table_numerics.index.astype(str)  # Ensure Attribute is a string

    # Reset the index to make "Attribute" a column and reorder columns
    summary_table_numerics = summary_table_numerics.reset_index(drop=True)[['Attribute', 'min', 'max', 'mean', 'median', 'std']]
    
    # Ensure there is something to merge
    if summary_table_numerics.empty:
        print("Summary statistics table is empty.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid summary stats
    
    # Merge metadata_numeric with summary_table_numerics on 'Attribute'
    metadata_numeric_renewed = pd.merge(
        metadata_numeric[["Attribute", "DataType", "Missing values", "Target"]].astype({'Attribute': 'str'}),
        summary_table_numerics,
        on='Attribute',
        how='inner'
    )
    # Check if the target_feature is present and rename it accordingly
    metadata_numeric_renewed['Attribute'] = metadata_numeric_renewed['Attribute'].apply(
        lambda x: f"{x}(target)" if x == target_feature else x
    )
    
    # Return the final table, or an empty DataFrame if the merge failed
    if metadata_numeric_renewed.empty:
        print("No valid data after merging metadata with numeric summary.")
        return pd.DataFrame()  # Return an empty DataFrame if merge yields no data
    
    # Round numeric columns to 2 decimal places
    metadata_numeric_renewed[['min', 'max', 'mean', 'median', 'std']] = metadata_numeric_renewed[['min', 'max', 'mean', 'median', 'std']].round(2)
    
    return metadata_numeric_renewed[["Attribute", 'min', 'max', 'mean', 'median', 'std', "Missing values"]]



# Function to create DataTable with empty check
def create_data_table(df, table_id):
    if df.empty:
        columns = [{'name': 'No data available', 'id': 'no_data'}]
        data = [{'no_data': 'No data available'}]
    else:
        columns = [{'name': i, 'id': i} for i in df.columns]
        data = df.to_dict('records')
    
    return dt.DataTable(
        id=table_id,
        columns=columns,
        data=data,
        fixed_rows={'headers': True, 'data': 0},  # Fix headers
        sort_action='native',  # Enable sorting
        sort_mode='single',  # Allow only single-column sorting
        filter_action='native',
        style_header={
            "backgroundColor": "#f8f9fa",  # Light gray header background
            "fontWeight": "bold",
            "fontFamily": "Arial, sans-serif",
            "borderBottom": "1px solid #dee2e6",  # Border for header cells
        },
        style_cell={
            "textAlign": "left",
            "backgroundColor": "#ffffff",
            "borderBottom": "1px solid #dee2e6",  # Border for data cells
            "padding": "10px",  # Padding inside cells
            "minWidth": "80px",  # Reduced minWidth
            "width": "100px",  # Reduced width
            "maxWidth": "200px",  # Reduced maxWidth
            "fontFamily": "Arial, sans-serif",
            "textOverflow": "ellipsis",
            "fontSize": 12,  # Slightly larger fontSize for better readability
            "color": "#495057",  # Darker text color
        },
        style_table={
            "minHeight": "300px",  # Slightly increased minHeight
            "maxHeight": "500px",  # Slightly increased maxHeight
            "marginBottom": "20px",
            "fontFamily": "Arial, sans-serif",
            "overflowY": "auto",
            "border": "1px solid #dee2e6",  # Border around the entire table
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",  # Light shadow for a card effect
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Target} = true',
                },
                'backgroundColor': '#FFF9C4',  # Light red background
                'color': 'black',  # Text color
            }
        ],
        page_action="none"
    )
    
    
markdown_text = """
Feature importance is calculated using a Random Forest model, which measures how much each feature reduces the impurity (error) when splitting the data across decision trees. Features that result in larger reductions are assigned higher importance scores.
"""