import dash
import openml
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from flask_caching import Cache

import re

import sys
import os
sys.path.append(os.path.abspath("C:/Users/20203203/Documents/data science Y4/Bachelor End Project/BepGitHub/OpenML/openml.org/server/src"))
from caching import CACHE_DIR_ROOT, CACHE_DIR_FLASK, CACHE_DIR_DASHBOARD

# from GENERAL_DATA_INFORMATION.layout_1 import layout_1
from GENERAL_DATA_INFORMATION.layout_1_2 import create_layout_1
from GENERAL_DATA_INFORMATION.callbacks_1 import register_callbacks_1

from INDIVIDUAL_FEATURE_EXPLORATION.layout_4 import create_layout_4
# from INDIVIDUAL_FEATURE_EXPLORATION.layout_4 import layout_4
from INDIVIDUAL_FEATURE_EXPLORATION.callbacks_4 import register_callbacks_4

from FEATURE_RELATION_EXPLORATION.layout_5 import create_layout_5
# from FEATURE_RELATION_EXPLORATION.layout_5 import layout_5
from FEATURE_RELATION_EXPLORATION.callbacks_5 import register_callbacks_5

# TODO: Move to assets (Copied from Joaquin's react font)
font = [
    "Nunito Sans",
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "Roboto",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
    "Apple Color Emoji",
    "Segoe UI Emoji",
    "Segoe UI Symbol",
]


def create_dash_app(flask_app):
    """
    Create dash application
    :param flask_app: flask_app passed is the flask server for the dash app
    :return:
    """

    app = dash.Dash(__name__, server=flask_app, url_base_pathname="/dashboard/", external_stylesheets=[dbc.themes.BOOTSTRAP])
    cache = Cache(
        app.server,
        config={
            "CACHE_TYPE": "filesystem",
            "CACHE_DIR": CACHE_DIR_FLASK,
        },
    )
    app.config.suppress_callback_exceptions = True

    # Layout of the dashboard
    app.layout = dbc.Container([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='data-id'),
        dbc.Tabs([
            dbc.Tab(label='Preliminary Viewing of the Data', tab_id='preliminary'),
            dbc.Tab(label='Individual Feature exploration', tab_id='data-dictionary'),
            dbc.Tab(label='Discover Relationships', tab_id='discover-relationships'),
        ], id='tabs', active_tab='preliminary'),
        html.Div(id='content')
    ], fluid=True)
    
    
    # Update when URL gets changed
    @app.callback(
        Output('data-id', 'value'),
        [Input('url', 'pathname')]
    )
    def update_data_id(pathname):
        if pathname is not None:
            number_flag = any(c.isdigit() for c in pathname)
            if "dashboard/data" in pathname and number_flag:
                data_id = int(re.search(r"data/(\d+)", pathname).group(1))         
                return data_id
        return None

    @app.callback(
        Output('content', 'children'), 
        [Input('tabs', 'active_tab'), Input('data-id', 'value')]
    )
    def render_content(tab, data_id):
        if data_id is None:
            return html.Div("No data ID provided.")
        
        
        if tab == 'preliminary':
            return html.Div([
                html.H3('Preliminary Viewing of the Data'),
                dbc.Card([
                    dbc.CardBody([
                        create_layout_1(data_id)
                        # Add more graphs or components as needed
                    ])
                ]),
                # Add more cards or components as needed
            ])
            
        elif tab == 'data-dictionary':
            return html.Div([
                # html.H3('Individual feature exploration'),
                dbc.Card([
                    dbc.CardBody([
                        create_layout_4(data_id)
                        # Add more tables or components as needed
                    ])
                ]),
                # Add more cards or components as needed
            ])
        elif tab == 'discover-relationships':
            return html.Div([
                html.H3('Discover Relationships between Features'),
                dbc.Card([
                    dbc.CardBody([
                        create_layout_5(data_id)
                        # Add more graphs or components as needed
                    ])
                ]),
                # Add more cards or components as needed
            ])
    register_callbacks_1(app)
    register_callbacks_4(app)
    register_callbacks_5(app)
    return app

# Assuming you have a Flask app instance
from flask import Flask
flask_app = Flask(__name__)

# Create the Dash app
dash_app = create_dash_app(flask_app)

if __name__ == '__main__':
    flask_app.run(debug=True)
    # dash_app.run_server(host='0.0.0.0', port=805)