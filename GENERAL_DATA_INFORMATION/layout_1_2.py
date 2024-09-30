from dash import Dash, html, dash_table as dt, dcc
import dash_bootstrap_components as dbc

import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from config import font
from helpers_1 import general_data_summary_table, data_quality_summary_table, create_nominals_metadata_table, create_numerics_metadata_table, create_data_table, markdown_text, calculate_marks
from helpers import get_metadata

def create_layout_1(data_id):
    # Get the data
    metadata, data, name = get_metadata(data_id)
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    
    # Need to convert the data to a dataframe we can work with
    dataframe = data.get_data()[0]
    
    # Rearrange data frame
    # Get the list of columns
    cols = dataframe.columns.tolist()

    # Reorder the list to move the specified column to the first position
    cols.insert(0, cols.pop(cols.index(target_feature)))
    
    # dataframe = dataframe.rename(columns={target_feature: target_feature + "(target)"})

    # Reindex the DataFrame with the new order of columns
    dataframe = dataframe[cols]
    # dataframe = dataframe.head(50)
    
    # Layout with a slider and a search bar
    layout_1 = html.Div([
    dbc.Card(
        dbc.CardBody([
            html.H4(f"Data Preview {name}", className="card-title"),

            # Slider to choose how many rows to display
            html.Div([
                html.Label("Select number of rows to display:"),
                dcc.Slider(
                    id='row-slider',
                    min=0,
                    max=len(dataframe),  # Full length of the dataframe as max value
                    step=10,  # Step can be adjusted based on preference
                    value=100,  # Default value (first 100 rows)
                    marks=calculate_marks(len(dataframe)),  # Dynamically generated proportional marks
                ),
            ], style={'marginBottom': '20px'}),

            # # Here I can implement the option to dropdown the nominal features and later their values
            # html.H6(f"Filter nominal features"),
            # # Dropdown filters for nominal (categorical) features
            # html.Div([
            #     html.Div([
            #         html.Label(f"Filter by {feature}:"),
            #         dcc.Dropdown(
            #             id=f'dropdown-{feature}',
            #             options=[{'label': val, 'value': val} for val in dataframe[feature].unique()],
            #             multi=True,
            #             placeholder=f'Select {feature} values',
            #         )
            #     ], style={'marginBottom': '10px'})  # Add margin between dropdowns
            #     for feature in nominal_features
            # ], style={'marginBottom': '20px'}),

            # DataTable to display the filtered subset of the data
            dt.DataTable(
                id='table',
                columns=[
                    {"name": (target_feature + " (target)" if i == target_feature else i), "id": i} for i in dataframe.columns
                ],
                data=dataframe.head(50).to_dict('records'),  # Show the first 'initial_subset_size' rows initially
                fixed_rows={'headers': True, 'data': 0},  # Fix headers
                sort_action='native',  # Enable sorting
                sort_mode='single',  # Allow only single-column sorting
                filter_action='native',  # Allow native filtering
                page_action="none",  # Disable paging (we'll manage rows with the slider)
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
            )
        ]),
        style={'marginBottom': '30px'}
    ),
        
        dbc.Card(
            dbc.CardBody([
                html.H4("General data summary", style={"fontFamily": font, 'marginBottom': '30px'}),
                general_data_summary_table(data_id), 
            ]), 
            style={'marginBottom': '30px'}
        ),
        
        dbc.Card(
            dbc.CardBody([
                html.H4("Data quality summary", style={"fontFamily": font, 'marginBottom': '30px'}),
                data_quality_summary_table(data_id),
            ]),
            style={'marginBottom': '30px'}
        ), 
        
        dbc.Card(
            dbc.CardBody([
                html.Div([
                html.H4("Feature Importance", style={"fontFamily": font, 'marginBottom': '30px'}),
                html.P(markdown_text),
                html.H5('Select Number of Most Important Features', style={'margin-bottom': '10px'}),  # Title for the input
                html.H6(f'There are a total of {dataframe.shape[1]} features', style={'margin-bottom': '10px'}),  # Title for the input

                dcc.Input(
                    id='nr-important-features',
                    type='number',
                    min=1,
                    max=dataframe.shape[1],
                    value=min(30, dataframe.shape[1]),
                    style={
                        'width': '100px',  # Set a fixed width
                        'padding': '5px',  # Add padding for a better look
                        'border-radius': '5px',  # Rounded corners
                        'border': '1px solid #ccc',  # Light gray border
                        'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'  # Light shadow for depth
                    }
                ),
                # feature_importance()[0],
                # dcc.Graph(id="fi"),
                # dcc.Graph(
                #     id='importance'
                # ),
                html.Div(
                    id="Feature Importance",
                    children=[
                        dcc.Loading(html.Div(id="fi")),
                        ],
                    )
                ])
                    ]),style={'marginBottom': '30px'}
        ), 
        
        dbc.Card(
            dbc.CardBody([
                html.H4("Metadata nominal features", className="card-title"), 
                create_data_table(create_nominals_metadata_table(data_id), 'metadata_nominals'),
            ]), style={'marginBottom': '30px'}, 
        ),
        
        dbc.Card(
            dbc.CardBody([
                html.H4("Metadata numeric features", className="card-title"), 
                create_data_table(create_numerics_metadata_table(data_id), 'metadata_numerics')
            ]), style={'marginBottom': '30px'}
        )
    ])
    
    return layout_1
    