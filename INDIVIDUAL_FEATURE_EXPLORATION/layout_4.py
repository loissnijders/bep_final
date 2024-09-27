from dash import dcc, html

import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# from load_data import dataframe, metadata
from helpers import get_metadata


def create_layout_4(data_id):
    metadata, data, name = get_metadata(data_id)
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    # type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]

    # Need to convert the data to a dataframe we can work with
    dataframe = data.get_data()[0]
    
    layout_4 = html.Div([
        dcc.Dropdown(
            id="feature_dropdown",
            options=[{'label': col, 'value': col} for col in dataframe.columns],
            value=[target_feature],
            multi=True
        ), 
        # html.Div(
        #     id='graph-container-distributions',
        #     style={"overflowY": "scroll", 'height': '700px'}
        # ),
        html.Div(
            id="Individual Distribution",
            children=[
                dcc.Loading(html.Div(id="graph-container-distributions")),
                ],
            )
    ])
    
    return layout_4