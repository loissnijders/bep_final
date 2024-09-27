from dash import dcc, html, dash_table as dt
import dash_bootstrap_components as dbc
import dash_daq as daq
from random import random

import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from config import font
from helpers import get_metadata
from helpers_5 import entropy_plot, check_nominal_target, heatmap_correlation, get_default_nom_val, feature_table, entropy_explanation

# Based on what kind of dataset there is, the layout gets chosen

def create_layout_5(data_id):
    metadata, data, name = get_metadata(data_id)
    dataframe = data.get_data()[0]
    
    nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    
    target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
    type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]


    if (len(numeric_features) == 0) and (len(nominal_features) > 0):
        
        layout_5 = html.Div(children=[
            dbc.Card(
                dbc.CardBody([
                    html.H1("(High dimensional) Dataset Exploration", style={"fontFamily": font}),
                    dcc.Graph(id="heatmap-correlation-figure-nominals"),
                ])
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H1(
                            f"Relationships between features",
                            style={"fontFamily": font},
                        ),
                    html.H3(
                            f"Relationship with target: {target_feature}",
                            style={"fontFamily": font},
                        ),
                    html.Label("Select features"),
                    dcc.Dropdown(
                        id='datatable_selected',  # New dropdown for selecting categories
                        options=[i for i in metadata.Attribute],  # To be populated dynamically
                        multi=True,
                        placeholder="Select up to 8 categories",
                        value=[i for i in metadata.Attribute[metadata.Attribute != target_feature]][0:2]
                    ), 
                    
                    # dcc.Dropdown(
                    #     id='category_selection_dropdown_target',  # New dropdown for selecting categories
                    #     options=[],  # To be populated dynamically
                    #     multi=True,
                    #     placeholder="Select up to 8 categories",
                    #     value=[]
                    # ),
                    html.Div(
                        dcc.Dropdown(
                            id='category_selection_dropdown_target',
                            options=[],
                            multi=True,
                            placeholder="Select up to 8 categories",
                            value=[]
                        ),
                        id='category_dropdown_container',
                        style={'display': 'none'}  # Hide the container by default
                    ),
                    
                    html.Div(
                            id='graph-container-target-distributions',
                            # style = {"overflowY": "scroll", 'height': '400px'}
                        ),
                ]), style={'marginBottom': '30px'}),
            
            dbc.Card(
                dbc.CardBody([
                    html.H3(
                            "Relationships between nominal features",
                            style={"fontFamily": font},
                        ),
                    # Entropy plot
                    
                    html.H4("Nominal features heatmap"),
                    # Heatmap
                    html.Div([
                        html.Label('Select Nominal Variable 1:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='nom_var_1_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in nominal_features],
                            value=check_nominal_target(data_id)  # Default value should be a single value
                        )
                    ], style={'margin-bottom': '10px'}),

                    html.Div([
                        html.Label('Select Nominal Variable 2:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='nom_var_2_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in nominal_features],
                            value=[i for i in metadata.Attribute[(metadata.Attribute != target_feature) & (metadata.DataType == "nominal")]][0]  # Default value should be a single value
                        )
                    ], style={'margin-bottom': '10px'}),
                    # Entropy plot
                    dcc.Graph(id="heatmap-nominals"),
                    
                    html.H3('Entropy plot nominal features', style={"fontFamily": font}),
                    html.P(entropy_explanation),
                    html.Label("Choose to view the number of features with the maximum entropy:"),
                    dcc.Input(
                        id='entropy-max',
                        type='number',
                        min=1,
                        max=len(nominal_features),
                        value=min(50, len(nominal_features)),
                    ),
                    dcc.Graph(id="entropy-plot"),
        
            ])),
        ])
        
    elif (len(nominal_features) == 0) and (len(numeric_features) > 0):
        layout_5 = html.Div(children=[
            dbc.CardBody([
                    html.H1("(High dimensional) Dataset Exploration", style={"fontFamily": font}),
                    html.H3("Coordinate plots", style={"fontFamily": font}),
                    html.Div([
                        html.Label("Select features"),
                        dcc.Dropdown(
                            id='feature-dropdown-coordinate-plot',
                            options=[{'label': col, 'value': col} for col in list(dataframe.columns)],
                            value=list(dataframe.columns)[:20],  # Default selected features
                            multi=True,
                            searchable=True,  # Enables search functionality
                            placeholder="Select features...",
                            style={'width': '100%'}
                        )
                    ], style={'height': '200px'}),  # Adjust height as needed
                    html.Div([
                        html.Label("Select feature for color scale", style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='color-dropdown',
                            options=[{'label': col, 'value': col} for col in list(dataframe.columns)],
                            value=target_feature,  # Default color mapping
                            clearable=False
                        ),
                        html.Label("Select Classes (for Nominal Color Feature):", style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='class-dropdown',
                            options=[],  # To be populated dynamically
                            value=[],
                            multi=True
                        )
                    ], style={'marginBottom': '30px'}),
                    dcc.Graph(id='parallel-coordinates-plot'),
                    
                    html.H3("Heatmap of numerical features", style={"fontFamily": font}),
                    dcc.Graph(figure=heatmap_correlation(data_id), id="heatmap-correlation-figure-numericals"),
                    
                ]),
            
            
            dbc.Card(
                dbc.CardBody([                
                html.H1(
                    f"Relationships between features",
                    style={"fontFamily": font},
                ),
                html.H3(
                    f"Relationship with target: {target_feature}",
                    style={"fontFamily": font},
                ),
                dcc.Dropdown(
                    id='datatable_selected',  # New dropdown for selecting categories
                    options=[i for i in metadata.Attribute],  # To be populated dynamically
                    multi=True,
                    placeholder="Select up to 8 categories",
                    value=[i for i in metadata.Attribute[metadata.Attribute != target_feature]][0:2]
                ),
                html.Div(
                        id='graph-container-target-distributions-numeric',
                    ),
            ]), style={'marginBottom': '30px'}),
            
            dbc.Card(
                dbc.CardBody([ 
                html.H3(
                        "Relationships between numeric features",
                        style={"fontFamily": font},
                    ),
                
                # Scatter plots 2 numerical variables
                html.Div([
                    html.Label('Select Numerical Variable 1:', style={"fontFamily": font}),
                    dcc.Dropdown(
                            id='num_var_1_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in numeric_features],
                            value=numeric_features[0]  # Default value should be a single value
                        )
                ], style={'margin-bottom': '10px'}),

                html.Div([
                    html.Label('Select Numerical Variable 2:', style={"fontFamily": font}),
                    dcc.Dropdown(
                        id='num_var_2_dropdown',
                        options=[{'label': cls, 'value': cls} for cls in numeric_features],
                        value=numeric_features[1]  # Default value should be a single value
                    )
                ], style={'margin-bottom': '10px'}),
                
                dcc.Graph(
                    id='basic-scatter-plot-numerics',
                ),
                                
                html.H3('Scatterplot matrix numeric features', style={"fontFamily": font}),
                    html.Label('Make a selection of numeric features', style={"fontFamily": font}),
                    dcc.RadioItems(
                        id="feature_choice_scatterplot_matrix",
                        options=["Top feature importance features", "Highest correlation", "Manual selection"],
                        value='Top feature importance features'
                    ),
                    dcc.Dropdown(
                        id='manual_numerics_scatterplot_matrix',
                        options=[{'label': cls, 'value': cls} for cls in numeric_features],
                        value=None,
                        style={'margin-bottom': '30px'},
                        multi=True
                    ),
                    html.Label('Select size of matrix', style={"fontFamily": font}),
                    dcc.Input(
                        id='size-scatterplot-matrix-numerics',
                        type='number',
                        min=1,
                        max=len(numeric_features),
                        value=min(4, len(numeric_features)),
                        style={
                            'width': '100px',  # Set a fixed width
                            'padding': '5px',  # Add padding for a better look
                            'border-radius': '5px',  # Rounded corners
                            'border': '1px solid #ccc',  # Light gray border
                            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'  # Light shadow for depth
                        }
                    ),
                    dcc.Graph(id="pairplots-scatter-plots-numerics"),
                ]))
            ])
        
    elif (len(numeric_features) > 0) and (len(nominal_features) > 0):
        layout_5 = html.Div(children=[
            dbc.Card(
                dbc.CardBody([
                    html.H1("(High dimensional) Dataset Exploration", style={"fontFamily": font}),
                    html.H3("Coordinate plots", style={"fontFamily": font}),
                    html.Div([
                        html.Label("Select features"),
                        dcc.Dropdown(
                            id='feature-dropdown-coordinate-plot',
                            options=[{'label': col, 'value': col} for col in list(dataframe.columns)],
                            value=list(dataframe.columns)[:20],  # Default selected features
                            multi=True,
                            searchable=True,  # Enables search functionality
                            placeholder="Select features...",
                            style={'width': '100%','marginBottom': '10px'}
                        )
                    ], style={'height': '200px','marginBottom': '10px'}),  # Adjust height as needed
                    html.Div([
                        html.Label("Select feature for color scale", style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='color-dropdown',
                            options=[{'label': col, 'value': col} for col in list(dataframe.columns)],
                            value=target_feature,  # Default color mapping
                            clearable=False
                        ),
                        html.Label("Select Classes (for Nominal Color Feature):", style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='class-dropdown',
                            options=[],  # To be populated dynamically
                            value=[],
                            multi=True
                        )
                    ], style={'marginBottom': '30px'}),
                    dcc.Graph(id='parallel-coordinates-plot'),
                    
                    html.H3("Heatmap of numerical features", style={"fontFamily": font}),
                    dcc.Graph(figure=heatmap_correlation(data_id), id="heatmap-correlation-figure-numericals"),
                    
                    html.H3("Heatmap of nominal features", style={"fontFamily": font}),
                    html.P("Add information on how this correlation is calculated"),
                    dcc.Graph(id="heatmap-correlation-figure-nominals"),
                ]),
                 
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H1(
                        f"Relationships between features",
                        style={"fontFamily": font},
                    ),
                    html.H3(
                        f"Relationship with target: {target_feature}",
                        style={"fontFamily": font},
                    ),
                    html.Label('Select a variable', style={"fontFamily": font}),
                    dcc.Dropdown(
                        id='datatable_selected',  # New dropdown for selecting categories
                        options=[i for i in metadata.Attribute],  # To be populated dynamically
                        multi=True,
                        placeholder="Select up to 8 categories",
                        value=[i for i in metadata.Attribute[metadata.Attribute != target_feature]][0:2]
                    ),
                    dcc.Dropdown(
                        id='category_selection_dropdown_target',  # New dropdown for selecting categories
                        options=[],  # To be populated dynamically
                        multi=True,
                        placeholder="Select up to 8 categories",
                        value=[]
                    ),
                    html.Div(id='graph-container-target-distributions'),
                ]), style={'marginBottom': '30px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H3("Relationships between numeric features", style={"fontFamily": font}),
                    html.Div([
                        html.Label('Select Numerical Variable 1:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='num_var_1_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in numeric_features],
                            value=numeric_features[0]  # Default value should be a single value
                        )
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Select Numerical Variable 2:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='num_var_2_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in numeric_features],
                            value=numeric_features[0]  # Default value should be a single value
                        )
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Select Nominal Variable:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='nom_val_dropdown',
                            options=[{'label': 'None', 'value': 'None'}] + [{'label': cls, 'value': cls} for cls in nominal_features],
                            value=get_default_nom_val(data_id)  # Default value
                        )
                    ], style={'margin-bottom': '10px'}),
                    dcc.Graph(id='basic-scatter-plot'),
                    html.H3('Scatterplot matrix numeric features', style={"fontFamily": font}),
                    html.Label('Make a selection of numeric features', style={"fontFamily": font}),
                    dcc.RadioItems(
                        id="feature_choice_scatterplot_matrix",
                        options=["Top feature importance features", "Highest correlation", "Manual selection"],
                        value='Top feature importance features'
                    ),
                    dcc.Dropdown(
                        id='manual_numerics_scatterplot_matrix',
                        options=[{'label': cls, 'value': cls} for cls in numeric_features],
                        value=None,
                        style={'margin-bottom': '30px'},
                        multi=True
                    ),
                    html.Label('Select size of matrix', style={"fontFamily": font}),
                    dcc.Input(
                        id='size-scatterplot-matrix-numerics',
                        type='number',
                        min=1,
                        max=len(numeric_features),
                        value=min(4, len(numeric_features)),
                        style={
                            'width': '100px',  # Set a fixed width
                            'padding': '5px',  # Add padding for a better look
                            'border-radius': '5px',  # Rounded corners
                            'border': '1px solid #ccc',  # Light gray border
                            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'  # Light shadow for depth
                        }
                    ),
                    dcc.Graph(id="pairplots-scatter-plots-numerics"),
                ]), style={'marginBottom': '30px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H3("Relationships between nominal features", style={"fontFamily": font}),
                    html.Div([
                        html.Label('Select Nominal Variable 1:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='nom_var_1_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in nominal_features],
                            value=check_nominal_target(data_id)  # Default value should be a single value
                        )
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Select Nominal Variable 2:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='nom_var_2_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in nominal_features],
                            value=check_nominal_target(data_id)  # Default value should be a single value
                        )
                    ], style={'margin-bottom': '10px'}),
                    dcc.Graph(id="heatmap-nominals"),
                    
                    html.H3('Entropy plot nominal features', style={"fontFamily": font}),
                    html.P(entropy_explanation),
                    html.Label("Choose to view the number of features with the maximum entropy:"),
                    dcc.Input(
                        id='entropy-max',
                        type='number',
                        min=1,
                        max=len(nominal_features),
                        value=max(20, len(nominal_features)),
                    ),
                    dcc.Graph(id="entropy-plot"),
                ]), style={'marginBottom': '30px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H3("Relationships between numeric and nominal features", style={"fontFamily": font}),
                    html.Div([
                        html.Label('Select Nominal Variable:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='nom_var_3_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in nominal_features],
                            value=get_default_nom_val(data_id)  # Default value should be a single value
                        ),
                        html.Label('Select Numerical Variable:', style={"fontFamily": font}),
                        dcc.Dropdown(
                            id='num_var_3_dropdown',
                            options=[{'label': cls, 'value': cls} for cls in numeric_features],
                            value=numeric_features[0]  # Default value should be a single value
                        ),
                        html.Label('Select order', style={"fontFamily": font})
                    ], style={'margin-bottom': '10px'}),
                    html.Label('Select the categories in the nominal variable', style={"fontFamily": font}),
                    dcc.Dropdown(
                        id='category_selection_dropdown',  # New dropdown for selecting categories
                        options=[],  # To be populated dynamically
                        multi=True,
                        placeholder="Select up to 8 categories",
                        value=[]
                    ),
                    dcc.Graph(id="categorical_boxplot"),
                ])
            ),
        ])

        
    return layout_5