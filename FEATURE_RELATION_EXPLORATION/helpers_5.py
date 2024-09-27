import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
from dash import dash_table as dt

import sys
import os
import re
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from config import font
# from load_data import dataframe, metadata, target_feature, type_target_feature
from helpers import get_metadata, clean_dataset

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import chi2_contingency

# nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
# numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)

###########################################################################################
# Checks

def check_nominal_target(data_id):
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]

    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
    
    if len(nominal_features) == 0:
            standard_nominal = ''
            
    else:
        if type_target_feature == "nominal":
            standard_nominal = target_feature
        
        else: 
            standard_nominal = None
    
    return standard_nominal

# Function to get the default value for the nominal dropdown
def get_default_nom_val(data_id):
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]

    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
    
    if len(nominal_features) == 0:
        return ''
    else:
        return check_nominal_target(data_id) if check_nominal_target(data_id) else nominal_features[0]


# Check if there are numerical features in the data

###########################################################################################
# Other helper functions

def calculate_entropy(series):
    value_counts = series.value_counts(normalize=True)
    return entropy(value_counts, base=2)

# Creating a frequency table
def create_frequency_table(df, cols):
    # Convert the dataframe into a list of tuples
    tuples = [tuple(x) for x in df[cols].values]
    # Create a counter dictionary
    freq_table = Counter(tuples)
    return freq_table

# Create a jitter function to avoid overlap
def jitter(values, jitter_amount=0.2):
    return values + np.random.uniform(-jitter_amount, jitter_amount, size=values.shape)

###########################################################################################


def heatmap_correlation(data_id, threshold=0):
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]

    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
    # Calculate correlation matrix
    df_numeric_features = dataframe[numeric_features]

    corr_matrix = df_numeric_features.corr()
    
    # Apply threshold: Keep only correlations above the threshold
    corr_matrix_thresholded = corr_matrix.where(abs(corr_matrix) >= threshold)

    # Remove rows and columns with all NaN values
    corr_matrix_filtered = corr_matrix_thresholded.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Keep only the upper triangle
    mask = np.triu(np.ones_like(corr_matrix_filtered, dtype=bool), k=1)
    upper_triangle = corr_matrix_filtered.where(mask)

    # Remove rows and columns with all NaN values again
    upper_triangle = upper_triangle.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Create heatmap
    fig2 = go.Figure(data=go.Heatmap(
        z=upper_triangle.values,
        x=upper_triangle.columns,
        y=upper_triangle.index,
        colorscale='Viridis',
        hoverongaps=False
    ))

    # Update layout
    fig2.update_layout(
        title='Upper Triangle Correlation Matrix Heatmap',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )

    return fig2



def entropy_plot(data_id):
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]

    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
    entropies = dataframe[nominal_features].apply(calculate_entropy)
    entropies = entropies.sort_values(ascending=False).round(3)
    
    # Create the plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=entropies.index,
        y=entropies.values,
        text=entropies.values,
        textposition='auto',
        marker_color='steelblue'  # Change the bar color to blue
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Entropy of Nominal Features',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Features',
        yaxis_title='Entropy',
        template='plotly_white',
        yaxis=dict(
            tickformat=".2f"  # Format y-axis ticks to two decimal places
        ),
        margin=dict(l=40, r=40, t=80, b=150)  # Adjust margins for better spacing
    )

    return fig


def feature_table(data_id):
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]

    # The table with all the features and their types to choose from
    feature_table= dt.DataTable(
        data = metadata[["Attribute", "DataType"]].to_dict('records'),
        columns = [{"name": i, "id": i} for i in metadata[["Attribute", "DataType"]].columns],
        row_selectable="multi",
        row_deletable=False,
        selected_rows=list(dataframe.columns)[:1], 
        id="datatable_selected",
        style_header={"backgroundColor": "white", "fontWeight": "bold"},
        style_cell={
                "textAlign": "left",
                "backgroundColor": "white",
                "minWidth": "100px",
                "width": "150px",
                "maxWidth": "300px",
                "fontFamily": font,
                "textOverflow": "ellipsis",
                "fontSize": 11,
            },
        style_table={
                "minHeight": "250px",
                "maxHeight": "250px",
                "marginBottom": "20px",
                "overflowY": "scroll",
            },
            page_action="none"
    )
    return feature_table

########################################################################################
def find_most_important_features(pathname, top_x):
    if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
        
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]
    
    try:
        target_attribute = metadata[metadata["Target"] == "true"]["Attribute"].values[0]
        target_type = metadata[metadata["Target"] == "true"]["DataType"].values[0]
    except IndexError:
        return "No target found", "No target found"

    # Feature importance bar plot
    from category_encoders.target_encoder import TargetEncoder

    df = dataframe
    x = df.drop(target_attribute, axis=1)
    y = df[target_attribute]

    te = TargetEncoder()
    if target_type == "nominal" or target_type == "string":
        y = pd.Categorical(y).codes
        x = clean_dataset(x)
        x = te.fit_transform(x, y)
        rf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0)
        rf.fit(x, y)
    else:
        x = clean_dataset(x)
        x = te.fit_transform(x, y)
        rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=0)
        rf.fit(x, y)

    fi = pd.DataFrame(
        rf.feature_importances_, index=x.columns, columns=["importance"]
    )
    fi = fi.sort_values("importance", ascending=False).reset_index()

    
    return fi["index"].head(top_x)

########################################################################################
def find_highest_correlated_numerics(pathname, size):
    if pathname:
        # Extract the ID number from the URL
        match = re.search(r'/(\d+)$', pathname)
        if match:
            data_id = int(match.group(1))  # Convert to integer

    # Get the data and metadata
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]

    # Get the numeric features from the metadata
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    numeric_df = dataframe[numeric_features]

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Convert the correlation matrix to a long format
    correlation_pairs = correlation_matrix.unstack()

    # Remove self-correlations (correlation of a feature with itself)
    correlation_pairs = correlation_pairs[correlation_pairs.index.get_level_values(0) != correlation_pairs.index.get_level_values(1)]

    # Sort the correlations by absolute value
    top_correlations = correlation_pairs.abs().sort_values(ascending=False)

    # Collect unique features from the top correlated pairs
    unique_features = set()
    for (feature1, feature2), correlation in top_correlations.items():
        unique_features.add(feature1)
        unique_features.add(feature2)
        if len(unique_features) >= size:
            break

    # Return exactly 'size' number of features
    return list(unique_features)[:size]


########################################################################################
# def create_parallel_coordinate_plot():
#     px.parallel_coordinates(df, color='target_feature')

# Is for the nominals 
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denominator = min((kcorr-1), (rcorr-1))
    if denominator == 0:
        return np.nan
    else:
        return np.sqrt(phi2corr / denominator)
    
    
entropy_explanation = "Entropy for nominal features in a dataset measures the amount of uncertainty or randomness in the distribution of categorical values. It quantifies how mixed or pure the distribution is—lower entropy indicates a more homogeneous distribution (less uncertainty), while higher entropy reflects a more diverse set of values (more uncertainty)."
    