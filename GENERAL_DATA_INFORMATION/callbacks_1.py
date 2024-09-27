# this is going to be the file where I'll define the callbacks related to section 1. GENERAL DATA INFORMATION

# the current version (14-06) doesn't make use of any callback hence this file is not used for now

import dash
from dash import dash_table as dt
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

from helpers import get_metadata
import re

from helpers import clean_dataset

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.graph_objects as go
import pandas as pd

def register_callbacks_1(app):
    # Dash callback to update the table rows
    @app.callback(
    Output('table', 'data'),
    [Input('row-slider', 'value'), Input('url', 'pathname')]
    )
    def update_table(row_slider, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        return dataframe.head(row_slider).to_dict('records')
    
    @app.callback(
        Output("fi", "children"),
        [Input("nr-important-features", "value"), Input('url', 'pathname')]
    )
    def feature_importance(nr_features, pathname):
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
        
        # Determine the number of features to include in the plot
        num_features_to_include = min(nr_features, dataframe.shape[1])
        
        fi_limited = fi.head(num_features_to_include)

        # Adjust the height based on the number of features to display
        bar_height = 20  # Height per bar in pixels
        total_height = max(500, nr_features * bar_height)  # Set a minimum height of 500 pixels

        trace = go.Bar(y=fi_limited["index"], x=fi_limited["importance"], name="fi", orientation="h")
        layout = go.Layout(
            autosize=False, margin={"l": 100, "t": 0}, height=total_height, hovermode="closest", 
            xaxis=dict(title=dict(text="Feature Importance Score", font=dict(size=14), standoff=10
        )  # Adding x-axis title
            )
        )
        figure = go.Figure(data=[trace], layout=layout)


        return html.Div(dcc.Graph(figure=figure), className="twelve columns")