from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px

import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from config import font

# from plot_functions import update_basic_scatter_plot, update_pairwise_scatter_plot, update_joint_plot, update_mutual_information_heatmap, update_entropy_plot, update_mosaic_plot, update_categorical_boxplots, update_swarmplot
from helpers_5 import check_nominal_target, heatmap_correlation, calculate_entropy, create_frequency_table, jitter, find_highest_correlated_numerics, find_most_important_features, cramers_v
from helpers import get_metadata

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import dash_daq as daq
import re

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

from scipy.stats import entropy

def register_callbacks_5(app):
    @app.callback(
        Output('graph-container-target-distributions', 'children'),
        [Input('datatable_selected', 'value'), 
         Input('category_selection_dropdown_target', 'value'),
         Input('url', 'pathname')] 
    )
    def update_graph(datatable_selected, selected_categories, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer

        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        # Get the data from the dataframe
        figs = []

        target_feature = metadata[metadata["Target"] == "true"]["Attribute"].values[0]
        type_target_feature = metadata[metadata["Target"] == "true"]["DataType"].values[0]

        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)

        for selected_attribute_name in datatable_selected:
            # Set a default value for selected_attribute_type
            selected_attribute_type = None

            if selected_attribute_name in nominal_features:
                selected_attribute_type = "nominal"
            elif selected_attribute_name in numeric_features:
                selected_attribute_type = "numeric"

            if (selected_attribute_type == 'numeric') and (type_target_feature == 'numeric'):
                fig = update_joint_plot(selected_attribute_name, target_feature, pathname)

            elif (selected_attribute_type == 'numeric') and (type_target_feature == 'nominal'):
                fig = update_categorical_boxplots(target_feature, selected_attribute_name, selected_categories, pathname)

            elif (selected_attribute_type == 'nominal') and (type_target_feature == 'numeric'):
                fig = update_categorical_boxplots(selected_attribute_name, target_feature, selected_categories, pathname)

            elif (selected_attribute_type == 'nominal') and (type_target_feature == 'nominal'):
                fig = update_mosaic_plot(selected_attribute_name, target_feature, pathname)

            figs.append(dcc.Graph(figure=fig))

        return figs
    
    
    @app.callback(
        Output('graph-container-target-distributions-numeric', 'children'),
        [Input('datatable_selected', 'value'), 
         Input('url', 'pathname')] 
    )
    def update_graph(datatable_selected, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
                
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        # Get the data from the dataframe
        figs = []

        target_feature = metadata[metadata["Target"] == "true"]["Attribute"].values[0]
        type_target_feature = metadata[metadata["Target"] == "true"]["DataType"].values[0]

        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)

        for selected_attribute_name in datatable_selected:
            # Set a default value for selected_attribute_type

            fig = update_joint_plot(selected_attribute_name, target_feature, pathname)


            figs.append(dcc.Graph(figure=fig))

        return figs
    
    # @app.callback(
    #     [Output('category_selection_dropdown_target', 'options'),
    #     Output('category_dropdown_container', 'style')],
    #     [Input('datatable_selected', 'value'), Input('url', 'pathname')]
    # )
    # def update_category_selection_dropdown(datatable_selected_values, pathname):
    #     if pathname:
    #         # Extract the ID number from the URL
    #         match = re.search(r'/(\d+)$', pathname)
    #         if match:
    #             data_id = int(match.group(1))  # Convert to integer
    #             print(f'Extracted ID: {data_id}')
    #         else:
    #             print('No ID found in the URL')

    #     metadata, data, name = get_metadata(data_id)
    #     dataframe = data.get_data()[0]
    #     # Get the data from the dataframe
    #     figs = []

    #     target_feature = metadata[metadata["Target"] == "true"]["Attribute"].values[0]
    #     type_target_feature = metadata[metadata["Target"] == "true"]["DataType"].values[0]

    #     nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
    #     numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
    #     options = []
    #     style = {'display': 'none'}  # Hide the dropdown by default

    #     # Check if the target feature is nominal
    #     if target_feature in nominal_features:
    #         categories = dataframe[target_feature].unique()
    #         options = [{'label': str(cat), 'value': cat} for cat in categories]
    #         style = {}  # Show the dropdown
    #     else:
    #         # Check if the selected feature(s) are nominal
    #         selected_nominal_vars = [var for var in datatable_selected_values if var in nominal_features] if datatable_selected_values else []
    #         if selected_nominal_vars:
    #             # For simplicity, use the first nominal variable
    #             var = selected_nominal_vars[0]
    #             categories = dataframe[var].unique()
    #             options = [{'label': str(cat), 'value': cat} for cat in categories]
    #             style = {}  # Show the dropdown

    #     return options, style
    
    #########################################################################################################################

    @app.callback(
    Output('class-dropdown', 'options'),
    Output('class-dropdown', 'value'),
    Output('class-dropdown', 'disabled'),
    Input('color-dropdown', 'value'),
    Input('url', 'pathname')
)
    def update_class_dropdown(color_feature, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        
        if color_feature in nominal_features:
            # Get unique categories
            df = dataframe.copy()
            df[color_feature] = df[color_feature].astype('category')
            categories = df[color_feature].cat.categories.tolist()
            options = [{'label': cat, 'value': cat} for cat in categories]
            return options, categories, False  # Enable dropdown
        else:
            return [], [], True  # Disable dropdown

    @app.callback(
        Output('parallel-coordinates-plot', 'figure'),
        Input('feature-dropdown-coordinate-plot', 'value'),
        Input('color-dropdown', 'value'),
        Input('class-dropdown', 'value'),
        Input('url', 'pathname')
    )
    def update_parallel_coordinates(selected_features, color_feature, selected_classes, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]

        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
        df = dataframe.copy()

        # Filter dataframe based on selected classes for nominal color feature
        if color_feature in nominal_features and selected_classes:
            df = df[df[color_feature].isin(selected_classes)]

        dimensions = []

        for col in selected_features:
            if col in nominal_features:
                # Handle categorical columns
                df[col] = df[col].astype('category')
                codes = df[col].cat.codes
                categories = df[col].cat.categories

                dimensions.append(
                    dict(
                        label=col,
                        values=codes,
                        tickvals=codes.unique(),
                        ticktext=categories,
                        range=[codes.min(), codes.max()]
                    )
                )
            elif col in numeric_features:
                # Handle numerical columns
                dimensions.append(
                    dict(
                        label=col,
                        values=df[col],
                        range=[df[col].min(), df[col].max()]
                    )
                )

        # Handle color mapping
        if color_feature in nominal_features:
            df[color_feature] = df[color_feature].astype('category')
            color_codes = df[color_feature].cat.codes
            color_categories = df[color_feature].cat.categories
            color = color_codes

            # Define a qualitative colorscale for categorical data
            num_categories = len(color_categories)
            colors = px.colors.qualitative.Set3  # Choose a qualitative colorscale with distinct colors
            colorscale = [[i / max(num_categories - 1, 1), colors[i % len(colors)]] for i in range(num_categories)]

            colorbar_tickvals = color_codes.unique()
            colorbar_ticktext = color_categories
        else:
            color = df[color_feature]
            colorscale = 'Viridis'
            colorbar_tickvals = None
            colorbar_ticktext = None

        # Create the parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=color,
                    colorscale=colorscale,
                    showscale=True,
                    cmin=color.min(),
                    cmax=color.max(),
                    colorbar=dict(
                        title=color_feature,
                        tickvals=colorbar_tickvals,
                        ticktext=colorbar_ticktext,
                    )
                ),
                dimensions=dimensions
            )
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig
    #########################################################################################################################
    
    @app.callback(
        Output('heatmap-correlation-figure-nominals', 'figure'),
        [Input('url', 'pathname'),
         Input("correlation-threshold-slider-nominals", "value")]
    )
    def update_nominal_heatmap(pathname, threshold):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
                
        # Compute Cramér's V matrix
        cramers_v_matrix = pd.DataFrame(index=nominal_features, columns=nominal_features, dtype=float)

        for col1 in nominal_features:
            for col2 in nominal_features:
                if col1 == col2:
                    cramers_v_matrix.loc[col1, col2] = 1.0
                else:
                    cramers_v_matrix.loc[col1, col2] = cramers_v(dataframe[col1], dataframe[col2])

        # Apply threshold
        cramers_v_matrix_thresholded = cramers_v_matrix.where(cramers_v_matrix >= threshold)


        # Remove rows and columns with all NaN values
        cramers_v_matrix_filtered = cramers_v_matrix_thresholded.dropna(axis=0, how='all').dropna(axis=1, how='all')

        if cramers_v_matrix_filtered.empty:
            # Handle empty matrix
            fig = go.Figure()
            fig.update_layout(
                title='No associations above the threshold.',
                xaxis={'visible': False},
                yaxis={'visible': False}
            )
            return fig

        # Compute total association for each feature
        total_association = cramers_v_matrix_filtered.sum(axis=1)

        # Order features by total association and limit to top N features
        top_features = total_association.sort_values(ascending=False).index

        # Filter the matrix to include only the top features
        cramers_v_matrix_filtered = cramers_v_matrix_filtered.loc[top_features, top_features]


        # Optionally keep only the upper triangle
        mask = np.triu(np.ones_like(cramers_v_matrix_filtered, dtype=bool), k=1)
        upper_triangle = cramers_v_matrix_filtered.where(mask)
        # Remove rows and columns with all NaN values again
        upper_triangle = upper_triangle.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # Prepare data for the heatmap
        z = upper_triangle.values
        x = upper_triangle.columns.tolist()
        y = upper_triangle.index.tolist()

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='Blues',
            colorbar=dict(title="Cramér's V"),
            zmin=0,
            zmax=1,
            hoverongaps=False
        ))

        # Update layout
        fig.update_layout(
            title="Association Heatmap for Nominal Features",
            xaxis_nticks=36,
            yaxis_nticks=36,
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=800,
            xaxis={'tickangle': -45}
        )
            
        return fig

    #########################################################################################################################
    @app.callback(
        Output('entropy-plot', 'figure'),
        [Input("entropy-max", "value"),
        Input('url', 'pathname')]
    )
    
    def update_entropy_plot(entropy_max, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
        
        target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
        type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
        entropies = dataframe[nominal_features].apply(calculate_entropy)
        entropies = entropies.sort_values(ascending=False).round(3)
        entropies = entropies.head(entropy_max)
        
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

    
        
    
    #########################################################################################################################

    @app.callback(
        Output('basic-scatter-plot', 'figure'),
        [Input("num_var_1_dropdown", "value"), 
        Input("num_var_2_dropdown", "value"),
        Input("nom_val_dropdown", "value"),
        Input('url', 'pathname')]
    )
    def update_basic_scatter_plot(num_var_1, num_var_2, nom_val, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        x = dataframe[num_var_1]
        y = dataframe[num_var_2]

        # Create scatter plot
        fig1 = go.Figure()
        
        # Means the user does not want to see the scatters class based
        if nom_val == "None":
            # Update layout
            fig1 = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))

            fig1.update_layout(
                title="Simple Scatter Plot",
                xaxis_title=num_var_1,
                yaxis_title=num_var_2
            )
            
        else:
            # Add data points to the plot for each class
            unique_classes = list(set(dataframe[nom_val]))
            for cls in unique_classes:
                cls_x = dataframe[dataframe[nom_val] == cls][num_var_1]
                cls_y = dataframe[dataframe[nom_val] == cls][num_var_2]
                fig1.add_trace(go.Scatter(x=cls_x, y=cls_y, mode='markers', name=cls, marker=dict(size=10)))

            # Update layout
            fig1.update_layout(
                title="Scatter Plot with Nominal Variable",
                xaxis_title=num_var_1,
                yaxis_title=num_var_2,
                legend_title=nom_val
            )

        return fig1

    @app.callback(
        Output('basic-scatter-plot-numerics', 'figure'),
        [Input("num_var_1_dropdown", "value"), 
        Input("num_var_2_dropdown", "value"),
        Input('url', 'pathname')]
    )
    def update_basic_scatter_plot(num_var_1, num_var_2, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        x = dataframe[num_var_1]
        y = dataframe[num_var_2]

        # Create scatter plot
        fig1 = go.Figure()
        
        # Means the user does not want to see the scatters class based
        # Update layout
        fig1 = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))

        fig1.update_layout(
            title="Simple Scatter Plot",
            xaxis_title=num_var_1,
            yaxis_title=num_var_2
        )

        return fig1

    ###########################################################################################


    @app.callback(
        Output('pairplots-scatter-plots', 'figure'),
        [Input('url', 'pathname'),
        Input("size_scatterplot_pair_matrix", "value"), 
        Input("nom_val_dropdown_2", "value")]
    )

    def update_pairwise_scatter_plot(pathname, num_variables=None, class_column=None):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        # Fetch metadata and data (assuming get_metadata is defined elsewhere)
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        # Get numeric features
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
        df_numeric_features = dataframe[numeric_features]
        df = df_numeric_features
        
        # Filter numeric columns
        numeric_columns = numeric_features
        
        # If num_variables is not specified, use all numeric columns
        if num_variables is None or num_variables > len(numeric_columns):
            num_variables = len(numeric_columns)
        
        # Select the first num_variables columns
        selected_columns = numeric_columns[:num_variables]
        
        # Create subplots
        fig3 = make_subplots(rows=num_variables, cols=num_variables, shared_xaxes=True, shared_yaxes=True)
        
        # Add scatter plots to the subplots
        for i, col1 in enumerate(selected_columns):
            for j, col2 in enumerate(selected_columns):
                if i != j:  # Only add scatter plot if it's not on the diagonal
                    fig3.add_trace(
                        go.Scatter(
                            x=df[col1], y=df[col2], mode='markers',
                            marker=dict(size=4),
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
        
        # Update layout for axes titles
        for i, col1 in enumerate(selected_columns):
            fig3.update_xaxes(title_text=col1, row=num_variables, col=i+1)
            fig3.update_yaxes(title_text=col1, row=i+1, col=1)
        
        # Update layout for the last row and column
        for i, col1 in enumerate(selected_columns):
            fig3.update_xaxes(title_text=selected_columns[i], row=num_variables, col=i+1)
            fig3.update_yaxes(title_text=selected_columns[i], row=i+1, col=1)
        
        # Update layout
        fig3.update_layout(height=300*num_variables, 
                        width=300*num_variables, 
                        title_text="Pairplot of Numeric Features (Empty Diagonal)",
                        showlegend=False
                        )
            
        # Show plot
        return fig3

#################################################################################

    @app.callback(
        Output('heatmap-correlation-figure-numericals', 'figure'),
        [Input('correlation-threshold-slider', 'value'),
         Input('url', 'pathname')]
    )
    def update_heatmap(threshold, pathname):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
        fig = heatmap_correlation(data_id, threshold)
        return fig



########
    @app.callback(
        Output('pairplots-scatter-plots-numerics', 'figure'),
        [Input('url', 'pathname'),
        Input("feature_choice_scatterplot_matrix", "value"),
        Input("size-scatterplot-matrix-numerics", "value"),
        Input("manual_numerics_scatterplot_matrix", "value")]
    )
    def update_pairwise_scatter_plot(pathname, feature_choice, size, manually_selected):
        if pathname:
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer

        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]

        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)

        size = min(size, len(numeric_features))

        if feature_choice == "Top feature importance features":
            selected_features = find_most_important_features(pathname, size)

        if feature_choice == "Highest correlation":
            selected_features = find_highest_correlated_numerics(pathname, size)
        
            
        if feature_choice == "Manual selection":
            selected_features = manually_selected
            size = len(manually_selected)

        # Define the maximum figure size that should fit on the page
        max_plot_size = 800  # Maximum dimension for the figure (in pixels)
        
        # Adjust the size of each subplot based on the number of features
        subplot_size = max_plot_size / size

        # Create a subplot grid
        fig = make_subplots(
            rows=size, cols=size,
            shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.02, vertical_spacing=0.02
        )

        for i, col_i in enumerate(selected_features):
            for j, col_j in enumerate(selected_features):
                if i == j:
                    # Diagonal: Add histograms
                    fig.add_trace(
                        go.Histogram(x=dataframe[col_i], nbinsx=20, showlegend=False),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: Add scatter plots
                    fig.add_trace(
                        go.Scatter(x=dataframe[col_j], y=dataframe[col_i], mode='markers',
                                marker=dict(size=3), showlegend=False),
                        row=i+1, col=j+1
                    )

                # Update x-axis titles only for the bottom row
                if i == size - 1:
                    fig.update_xaxes(title_text=col_j, row=i+1, col=j+1, tickangle=90)
                # Update y-axis titles only for the first column
                if j == 0:
                    fig.update_yaxes(title_text=col_i, row=i+1, col=j+1, tickangle=0)

        # Update the layout with the calculated figure size
        fig.update_layout(
            height=max_plot_size,  # Set the overall height
            width=max_plot_size,   # Set the overall width
            title="Scatterplot Matrix",
            showlegend=False
        )

        # Update axes to remove gridlines for cleaner plots
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig


    ###########################################################################################
    @app.callback(
        Output("mutual-information-heatmap", "figure"),
        [Input('mutual_information_choice', "value"),
        Input('url', 'pathname')]
    )
    def update_mutual_information_heatmap(mutual_information_choice, pathname):
        """ Plots the mutual information. If only_nominals = True, it only displays the heatmap of the nominal features. If false, it shows the
        mutual information heatmap of all of the features
        """
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
        
        if mutual_information_choice == "Nominal":
            # Apply label encoding to the nominal features
            df_encoded = dataframe.copy()
            le = LabelEncoder()
            for feature in nominal_features:
                df_encoded[feature] = le.fit_transform(df_encoded[feature])

            # Calculate mutual information between nominal features
            mi_nominal_matrix = pd.DataFrame(index=nominal_features, columns=nominal_features)

            # Calculate mutual information for each pair of nominal features
            for feature in nominal_features:
                mi = mutual_info_classif(df_encoded[nominal_features], df_encoded[feature])
                mi_nominal_matrix[feature] = mi

            # Convert the matrix to float for display purposes
            mi_nominal_matrix = mi_nominal_matrix.astype(float)
            
            # Create the Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=mi_nominal_matrix.values,
                x=mi_nominal_matrix.columns,
                y=mi_nominal_matrix.index,
                colorscale='Viridis',
                text=mi_nominal_matrix.values,
                hoverinfo='text'
            ))

            # Update the layout for better readability
            fig.update_layout(
                title='Mutual Information Heatmap of Nominal Features',
                xaxis_nticks=36,
                xaxis_tickangle=-45,
                yaxis_autorange='reversed'
            )
            
        else:
            # Apply label encoding to the nominal features
            df_encoded = dataframe.copy()
            le = LabelEncoder()
            for feature in nominal_features:
                df_encoded[feature] = le.fit_transform(df_encoded[feature])

            # Initialize the mutual information matrix
            all_features = numeric_features + nominal_features
            mi_matrix = pd.DataFrame(index=all_features, columns=all_features, dtype=float)

            # Calculate mutual information for numeric features
            for col1 in numeric_features:
                for col2 in numeric_features:
                    if col1 != col2:
                        mi = mutual_info_regression(dataframe[[col1]], dataframe[col2])
                        mi_matrix.loc[col1, col2] = mi[0]

            # Calculate mutual information for nominal features
            for col1 in nominal_features:
                for col2 in nominal_features:
                    if col1 != col2:
                        mi = mutual_info_classif(df_encoded[[col1]], df_encoded[col2])
                        mi_matrix.loc[col1, col2] = mi[0]

            # Calculate mutual information between numeric and nominal features
            kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            for col1 in numeric_features:
                transformed_col1 = kbin.fit_transform(dataframe[[col1]])
                for col2 in nominal_features:
                    mi = mutual_info_classif(transformed_col1, df_encoded[col2])
                    mi_matrix.loc[col1, col2] = mi[0]
                    mi_matrix.loc[col2, col1] = mi[0]

            # Fill diagonal with zeros as mutual information with self is not informative
            np.fill_diagonal(mi_matrix.values, 0)

            # Create the plotly heatmap
            heatmap = go.Heatmap(
                z=mi_matrix.values,
                x=mi_matrix.columns,
                y=mi_matrix.index,
                colorscale='Viridis',
                text=mi_matrix.values,
                hoverinfo='text'
            )

            layout = go.Layout(
                title='Mutual Information Between All Features',
                xaxis=dict(title='Features'),
                yaxis=dict(title='Features'),
                coloraxis=dict(colorbar=dict(title='Mutual Information'))
            )

            fig = go.Figure(data=[heatmap], layout=layout)
            
        return fig

    ###########################################################################################

    @app.callback(
        Output('mosaic-plot', 'figure'),
        [Input("nom_var_1_dropdown", "value"), 
        Input("nom_var_2_dropdown", "value"),
        Input('url', 'pathname')]
    )
    # Function to create a mosaic plot
    def update_mosaic_plot(variable1, variable2, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        df = dataframe
        # Create frequency table
        freq_table = create_frequency_table(df, [variable1, variable2])

        # Filter out zero counts
        freq_table = {k: v for k, v in freq_table.items() if v > 0}

        # Calculate proportions
        total = sum(freq_table.values())
        proportions = {k: v / total for k, v in freq_table.items()}

        # Calculate positions and sizes
        x_pos = 0
        rects = []
        for var1_value in df[variable1].unique():
            y_pos = 0
            var1_total = sum(proportions[(var1_value, var2_value)] for var2_value in df[variable2].unique() if (var1_value, var2_value) in proportions)
            for var2_value in df[variable2].unique():
                if (var1_value, var2_value) in proportions:
                    width = var1_total
                    height = proportions[(var1_value, var2_value)] / var1_total
                    rects.append((x_pos, y_pos, width, height, var1_value, var2_value))
                    y_pos += height
            x_pos += var1_total

        # Create the Plotly figure
        fig = go.Figure()

        for (x, y, width, height, var1_value, var2_value) in rects:
            fig.add_trace(go.Scatter(
                x=[x, x+width, x+width, x, x],
                y=[y, y, y+height, y+height, y],
                fill='toself',
                name=f'{var1_value} - {var2_value}',
                text=f'{var1_value} - {var2_value}: {width*height:.2%}',
                hoverinfo='text'
            ))

            # Add annotations
            fig.add_annotation(
                x=x + width / 2,
                y=y + height / 2,
                text=f'{var1_value}<br>{var2_value}',
                showarrow=False,
                font=dict(size=10, color="white"),
                align='center',
                xanchor='center',
                yanchor='middle'
            )

        fig.update_layout(
            title=f'Mosaic Plot of {variable1} and {variable2}',
            xaxis_title=variable1,
            yaxis_title=variable2,
            showlegend=True,
            legend_title=f'{variable1} - {variable2}'
        )

        return fig
    
    
###########################################################################################

    @app.callback(
        Output('heatmap-nominals', 'figure'),
        [Input("nom_var_1_dropdown", "value"), 
        Input("nom_var_2_dropdown", "value"),
        Input('url', 'pathname')]
    )
    # Function to create a mosaic plot
    def update_mosaic_plot(variable1, variable2, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        # Create a contingency table (cross-tabulation) of the two nominal variables
        contingency_table = pd.crosstab(dataframe[variable1], dataframe[variable2])

        # Plotly - from contingency table
        heatmap = go.Figure(data=go.Heatmap(
            z=contingency_table.values,  # Values (counts)
            x=contingency_table.columns,  # Categories of Category2
            y=contingency_table.index,    # Categories of Category1
            colorscale='Blues',  # Color scale (can be changed to others like 'Cividis', 'Blues', etc.)
            colorbar=dict(title="Frequency") 
        ))

        # Add titles and labels
        heatmap.update_layout(
            title=f'Heatmap of {variable1} vs {variable2}',
            xaxis_title=variable2,
            yaxis_title=variable1
        )

        return heatmap

    ###########################################################################################
    @app.callback(
        [Output('category_selection_dropdown', 'options'),  # Dynamically populate categories
         Output('category_selection_dropdown', 'value')],   # Reset value when new nominal feature is selected
        [Input('nom_var_3_dropdown', 'value'), Input('url', 'pathname')]
    )
    def update_category_dropdown(nominal_feature, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        if nominal_feature:
            # Assuming `dataframe` is globally available or passed in some other way
            nominal_classes = sorted(dataframe[nominal_feature].cat.categories)
            options = [{'label': category, 'value': category} for category in nominal_classes]
            
            return options, nominal_classes[:8]  # Automatically select first 8 categories by default
        return [], []
    
    ###########################################################################################
    @app.callback(
    Output('categorical_boxplot', 'figure'),
    [Input('nom_var_3_dropdown', 'value'), 
     Input('num_var_3_dropdown', 'value'),
     Input('category_selection_dropdown', 'value'),   # Selected categories
     Input('url', 'pathname')]
    )
    def update_categorical_boxplots(nominal_feature, numeric_feature, selected_categories, pathname):
        if pathname:
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))

        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        fig = go.Figure()
        
        # Get sorted categories and their counts
        nominal_classes_counts = dataframe[nominal_feature].value_counts()

        if selected_categories:
            # Ensure maximum of 8 categories
            selected_categories = selected_categories[:8]
            
            for category in selected_categories:
                count = nominal_classes_counts[category]
                fig.add_trace(go.Box(
                    x=dataframe[dataframe[nominal_feature] == category][numeric_feature],
                    y=[category] * len(dataframe[dataframe[nominal_feature] == category]),
                    name=f"{category} (n={count})",
                    orientation='h',
                    boxmean=True
                ))
        
        fig.update_layout(
            title=f"Boxplot of {numeric_feature} by {nominal_feature}",
            xaxis_title=numeric_feature,
            yaxis_title=nominal_feature,
            showlegend=False,
            boxmode='group'
        )
        
        return fig
    ###########################################################################################
    
    
    
    
    
    
    
    
    # Create similar code for boxplot relationship with target
    @app.callback(
        [Output('category_selection_dropdown_target', 'options'),  # Dynamically populate categories
         Output('category_selection_dropdown_target', 'value')],   # Reset value when new nominal feature is selected
        [Input('datatable_selected', 'value'), Input('url', 'pathname')]
    )
    def update_category_dropdown_target(datatable_selected, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
        type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]
        
        nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
        numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)
        
        
        if type_target_feature == "nominal":
                nominal_classes = sorted(dataframe[target_feature].cat.categories)
                options = [{'label': category, 'value': category} for category in nominal_classes]
                
                return options, nominal_classes[:8]  # Automatically select first 8 categories by default
            
        else:
            for feature in datatable_selected:
                if feature in nominal_features:
                    nominal_classes = sorted(dataframe[feature].cat.categories)
                    options = [{'label': category, 'value': category} for category in nominal_classes]
                
                    return options, nominal_classes[:8]  # Automatically select first 8 categories by default
    
    # ###########################################################################################
    def update_categorical_boxplots_target(nominal_feature, numeric_feature, selected_categories_target, pathname):
        if pathname:
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))

        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        fig = go.Figure()
        
        # Get sorted categories and their counts
        nominal_classes_counts = dataframe[nominal_feature].value_counts()

        if selected_categories_target:
            # Ensure maximum of 8 categories
            selected_categories_target = selected_categories_target[:8]
            
            for category in selected_categories_target:
                count = nominal_classes_counts[category]
                fig.add_trace(go.Box(
                    x=dataframe[dataframe[nominal_feature] == category][numeric_feature],
                    y=[category] * len(dataframe[dataframe[nominal_feature] == category]),
                    name=f"{category} (n={count})",
                    orientation='h',
                    boxmean=True
                ))
        
        fig.update_layout(
            title=f"Boxplot of {numeric_feature} by {nominal_feature}",
            xaxis_title=numeric_feature,
            yaxis_title=nominal_feature,
            showlegend=False,
            boxmode='group'
        )
        
        return fig
    
    
    
    ###########################################################################################

    @app.callback(
        Output('swarmplot', 'figure'),
        [Input("nom_var_3_dropdown", "value"), 
        Input("num_var_3_dropdown", "value"),
        Input('url', 'pathname')]
    )
    def update_swarmplot(nominal_feature, numeric_feature, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        
        # Create a mapping from category to index
        category_indices = {category: idx for idx, category in enumerate(dataframe[nominal_feature].unique())}
        
        # Create the figure
        fig = go.Figure()

        # Add scatter plots for each category in the nominal feature
        for category, idx in category_indices.items():
            subset = dataframe[dataframe[nominal_feature] == category]
            fig.add_trace(go.Scatter(
                x=subset[numeric_feature],
                y=jitter(np.full_like(subset[numeric_feature], idx)),
                mode='markers',
                name=category
            ))

        # Update layout
        fig.update_layout(
            title=f"Swarmplot of {numeric_feature} by {nominal_feature}",
            xaxis_title=numeric_feature,
            yaxis_title=nominal_feature,
            yaxis=dict(
                tickmode='array',
                tickvals=list(category_indices.values()),
                ticktext=list(category_indices.keys())
            ),
            showlegend=False
        )

        return fig
    
    @app.callback(
        Output('joint_plot', 'figure'),
        [Input("num_var_1_dropdown", "value"), 
        Input("num_var_2_dropdown", "value"),
        Input('url', 'pathname')]
    )

    def update_joint_plot(selected_numeric_variable, target_variable, pathname):
        if pathname:
            # Extract the ID number from the URL
            # Extract the ID number from the URL
            match = re.search(r'/(\d+)$', pathname)
            if match:
                data_id = int(match.group(1))  # Convert to integer
            
        metadata, data, name = get_metadata(data_id)
        dataframe = data.get_data()[0]
        # Generate a joint plot
        fig = px.scatter(dataframe, x=selected_numeric_variable, y=target_variable, 
                        marginal_x='histogram', marginal_y='histogram')

        # Update layout for better visualization
        fig.update_layout(title=f'Joint Plot of {selected_numeric_variable} vs {target_variable}',
                        xaxis_title=selected_numeric_variable,
                        yaxis_title=target_variable)

        return fig