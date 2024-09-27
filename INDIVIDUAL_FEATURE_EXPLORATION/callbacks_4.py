import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import dash_daq as daq

import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from helpers import get_metadata
# from helpers_4 import numeric_features, nominal_features
import re
import ast


def register_callbacks_4(app):
    @app.callback(
        Output('graph-container-distributions', 'children'),
        [Input('feature_dropdown', 'value'), Input('url', 'pathname')]  
    )
    def update_content(feature_dropdown, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
                print(f'Extracted ID: {data_id}')
            else:
                print('No ID found in the URL')

        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]

        print("callback is triggered")

        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)

        content = []

        for feature in feature_dropdown:

            if feature in numeric_features:
                print(f"The feature {feature} is numeric")
                section_header = html.H3(f"Distribution of feature: {feature} (numeric)", style={'marginBottom': '10px', 'marginTop': '20px'})
                content.append(section_header)
                try:
                    # Create Histogram
                    fig = go.Figure(data=[go.Histogram(x=dataframe[feature])])
                    fig.update_layout(
                        title=f'Histogram of {feature}',
                        xaxis_title='Value',
                        yaxis_title='Frequency'
                    )
                    graph = dcc.Graph(figure=fig, id={'type': 'dynamic-graph-num', 'index': feature})

                    # Calculate Descriptive Statistics
                    desc = dataframe[feature].describe(percentiles=[.25, .5, .75]).to_frame().T
                    desc['Percentage of Missing Values'] = str(100 * dataframe[feature].isnull().mean()) + "%"
                    desc = desc.round(2)
                    
                    # Transpose Table
                    table_header = [html.Thead(html.Tr([html.Th("Statistic"), html.Th("Value")]))]
                    table_body = [html.Tbody([html.Tr([html.Td(stat), html.Td(desc.iloc[0][stat])]) for stat in desc.columns])]
                    table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)

                    # Combine graph and table in a row with controls above
                    row = html.Div(
                        style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': '20px'},
                        children=[
                            # Initialize here the number of bins selector
                            html.Div(
                                style={'marginBottom': '10px'},
                                children=[
                                    html.Label('Number of bins:'),
                                    dcc.Slider(
                                        id={'type': 'bin-slider', 'index': feature},
                                        min=1, max=50, step=1, value=30,  # Adjust the min, max, and default bin values as needed
                                        marks={i: str(i) for i in range(5, 51, 5)},  # Example marks every 5
                                    )
                                ]
                            ),
                            # Graph and Table Row
                            html.Div(
                                style={'display': 'flex', 'flexDirection': 'row'},
                                children=[
                                    html.Div(graph, style={'width': '70%', 'display': 'inline-block'}),
                                    html.Div(table, style={'width': '25%', 'display': 'inline-block'})
                                ]
                            )
                        ]
                    )
                    content.append(row)
                except Exception as e:
                    print(f'Error creating numeric feature graph and table for {feature}: {e}')

            elif feature in nominal_features:
                section_header = html.H3(f"Distribution of feature: {feature} (nominal)", style={'marginBottom': '10px', 'marginTop': '20px'})
                content.append(section_header)

                print(f"The feature {feature} is nominal")
                try:
                    # Get Value Counts
                    value_counts = dataframe[feature].value_counts()
                    num_distinct_classes = len(value_counts)

                    # Create Frequency Plot
                    fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
                    fig.update_layout(
                        title=f'Frequency Plot of {feature}',
                        xaxis_title='Category',
                        yaxis_title='Frequency'
                    )
                    graph = dcc.Graph(id={'type': 'dynamic-graph', 'index': feature}, figure=fig)

                    # Calculate Statistics for Nominal Feature
                    missing_count = dataframe[feature].isnull().sum()
                    missing_pct = str(100 * dataframe[feature].isnull().mean()) + "%"

                    # Create Statistics Table for Nominal Feature
                    stats = {
                        "Number of Distinct Classes": num_distinct_classes,
                        "Number of Missing Values": missing_count,
                        "Percentage of Missing Values": missing_pct,
                    }
                    table_header = [html.Thead(html.Tr([html.Th("Statistic"), html.Th("Value")]))]
                    table_body = [html.Tbody([html.Tr([html.Td(stat), html.Td(value)]) for stat, value in stats.items()])]
                    table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)

                    # Combine graph, table, and controls in a column layout
                    row = html.Div(
                        style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': '20px'},
                        children=[
                            # Controls Column
                            html.Div(
                                style={'display': 'flex', 'flexDirection': 'column', 'marginBottom': '10px'},
                                children=[
                                    html.H6("Select a subset of categories of this nominal feature"),
                                    dcc.RadioItems(
                                        id={'type': 'categories-subset', 'index': feature},
                                        options=[
                                            {'label': 'No subset, all categories', 'value': 'all_categories'},
                                            {'label': 'Most occurring', 'value': 'most_occurring'},
                                            {'label': 'Least occurring', 'value': 'least_occurring'},
                                            {'label': 'Manual subset selection', 'value': 'manual'},
                                        ],
                                        value='all_categories',  # Default option
                                        labelStyle={'display': 'block'},
                                    ),
                                    html.Div(id={'type': 'dynamic-choice', 'index': feature}),
                                    html.Div(
                                        id={'type': 'subset-control', 'index': feature},
                                        children=[
                                            # Dropdown for Manual Selection
                                            dcc.Dropdown(
                                                id={'type': 'manual-subset-dropdown', 'index': feature},
                                                options=[{'label': cat, 'value': cat} for cat in value_counts.index],
                                                multi=True,
                                                placeholder='Select categories manually',
                                                style={'marginTop': '10px', 'marginBottom': '30px'}
                                            ),
                                            
                                            
                                            # Numeric Input for Top/Least occurring
                                            html.Label("Indicate the number of most/least occurring classes"),
                                            dcc.Input(
                                                id={'type': 'numeric-input', 'index': feature},
                                                type='number',
                                                min=1,
                                                max=num_distinct_classes,
                                                value=num_distinct_classes,
                                                style={
                                                    'width': '100px',  # Set a fixed width
                                                    'padding': '5px',  # Add padding for a better look
                                                    'border-radius': '5px',  # Rounded corners
                                                    'border': '1px solid #ccc',  # Light gray border
                                                    'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)'  # Light shadow for depth
                                                }
                                            ),
                                        ]
                                    ),
                                    html.H6("Select view of the frequency plot"),
                                    dcc.RadioItems(
                                        id={'type': 'axis-sort', 'index': feature},
                                        options=[
                                            {'label': 'Sort by Frequency', 'value': 'frequency'},
                                            {'label': 'Sort Alphabetically/Numerically', 'value': 'alphabetical_numerical'},
                                        ],
                                        value='alphabetical_numerical',  # Default option
                                        labelStyle={'display': 'block'}
                                    )
                                ]
                            ),
                            # Graph and Table Row
                            html.Div(
                                style={'display': 'flex', 'flexDirection': 'row'},
                                children=[
                                    html.Div(graph, style={'width': '70%', 'display': 'inline-block'}),
                                    html.Div(table, style={'width': '25%', 'display': 'inline-block'})
                                ]
                            )
                        ]
                    )
                    content.append(row)
                except Exception as e:
                    print(f'Error creating nominal feature graph and table for {feature}: {e}')

        return content
    
    #######################################################################################################

    
    #######################################################################################################

    @app.callback(
        Output({'type': 'dynamic-graph', 'index': dash.dependencies.MATCH}, 'figure'),
        [
            Input({'type': 'numeric-input', 'index': dash.dependencies.MATCH}, 'value'),
            Input({'type': 'axis-sort', 'index': dash.dependencies.MATCH}, 'value'),
            Input({'type': 'categories-subset', 'index': dash.dependencies.MATCH}, 'value'),
            Input({'type': 'manual-subset-dropdown', 'index': dash.dependencies.MATCH}, 'value'),
            Input('feature_dropdown', 'value'),
            Input('url', 'pathname')
        ]
    )
    def update_nominal_plot(numeric_input_value, axis_sort, subset_type, manual_subset, feature_dropdown, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
                print(f'Extracted ID: {data_id}')
            else:
                print('No ID found in the URL')

        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]

        # Check if the feature dropdown has any selected features
        if not feature_dropdown:
            return dash.no_update

        # Use dash.callback_context to get which input triggered this callback
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Safely parse the input_id to extract the feature name
        try:
            feature = ast.literal_eval(input_id).get('index', None)
        except (ValueError, SyntaxError):
            print(f"Error parsing input ID: {input_id}")
            return dash.no_update

        if feature is None:
            print(f"No valid feature extracted from input ID: {input_id}")
            return dash.no_update

        try:
            # Get the value counts of the selected feature
            value_counts = dataframe[feature].value_counts()

            # Ensure the counts are numeric before applying nlargest/nsmallest
            value_counts = value_counts.astype(int)  # Convert counts to integers if not already
        

            # Handle the subset selection
            # Handle the subset selection
            if subset_type == 'most_occurring':
                top_values = value_counts.nlargest(numeric_input_value)
                other_values = value_counts[~value_counts.index.isin(top_values.index)].sum()
                combined_values = pd.concat([top_values, pd.Series({'Other': other_values})]) if other_values > 0 else top_values

            elif subset_type == 'least_occurring':
                least_values = value_counts.nsmallest(numeric_input_value)
                other_values = value_counts[~value_counts.index.isin(least_values.index)].sum()
                combined_values = pd.concat([least_values, pd.Series({'Other': other_values})]) if other_values > 0 else least_values

            elif subset_type == 'manual' and manual_subset:
                selected_values = value_counts[value_counts.index.isin(manual_subset)]
                other_values = value_counts[~value_counts.index.isin(manual_subset)].sum()
                combined_values = pd.concat([selected_values, pd.Series({'Other': other_values})]) if other_values > 0 else selected_values

            else:  # All categories
                combined_values = value_counts

            # Handle sorting
            if axis_sort == 'frequency':
                combined_values = combined_values.sort_values(ascending=False)
            elif axis_sort == 'alphabetical_numerical':
                combined_values = combined_values.sort_index()

            # Create updated frequency plot
            fig = go.Figure(data=[go.Bar(x=combined_values.index, y=combined_values.values)])
            fig.update_layout(
                title=f'Frequency Plot of {feature}',
                xaxis_title=feature,
                yaxis_title='Frequency'
            )

            return fig


        except Exception as e:
            print(f"Error updating nominal plot for feature {feature}: {e}")
            return dash.no_update


    @app.callback(
        Output({'type': 'dynamic-graph-num', 'index': dash.dependencies.MATCH}, 'figure'),
        [
            Input({'type': 'bin-slider', 'index': dash.dependencies.MATCH}, 'value'),
            Input('url', 'pathname')
        ]
    )
    def update_histogram(nbins, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
                print(f'Extracted ID: {data_id}')
            else:
                print('No ID found in the URL')
        
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        # Determine which feature's slider was triggered
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        feature = eval(input_id)['index']  # Get the feature name from the slider's ID
        
        # Update histogram with new number of bins
        fig = go.Figure(data=[go.Histogram(x=dataframe[feature], nbinsx=nbins)])
        fig.update_layout(
            title=f'Histogram of {feature}',
            xaxis_title='Value',
            yaxis_title='Frequency'
        )
        return fig
    
    
    
