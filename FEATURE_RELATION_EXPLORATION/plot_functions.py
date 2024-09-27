from load_data import dataframe, metadata, target_feature, type_target_feature
import plotly.graph_objs as go

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from scipy.stats import entropy
from statsmodels.graphics.mosaicplot import mosaic
from collections import Counter
from dash import dash_table as dt
from config import font
from plotly.subplots import make_subplots
import plotly.express as px

from helpers_5 import create_frequency_table, jitter, calculate_entropy

nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)


# Define here the functions that are used to create the visualizations
# This way, the callback file will be much shorter and more convenient 

# RELATIONS BETWEEN NUMERICS
# basic-scatter-plot
def update_basic_scatter_plot(num_var_1, num_var_2, nom_val):
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


# pairplots-scatter-plots

def update_pairwise_scatter_plot(num_variables=None, class_column=None):
    df_numeric_features = dataframe[numeric_features]
    df = df_numeric_features
    # Filter numeric columns
    
    numeric_columns = df.select_dtypes(include=np.number).columns
    
    # If num_variables is not specified, use all numeric columns
    if num_variables is None or num_variables > len(numeric_columns):
        num_variables = len(numeric_columns)
    
    # Select the first num_variables columns
    selected_columns = numeric_columns[:num_variables]
    
    # Create a color map for the classes
    if class_column:
        unique_classes = dataframe[class_column].unique()
        color_map = {cls: f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.8)' for cls in unique_classes}
    else:
        color_map = {}
    
    # Create subplots
    fig3 = make_subplots(rows=num_variables, cols=num_variables, shared_xaxes=True, shared_yaxes=True)
    
    # Add scatter plots to the subplots
    for i, col1 in enumerate(selected_columns):
        for j, col2 in enumerate(selected_columns):
            if i != j:
                for cls in unique_classes:
                    mask = dataframe[class_column] == cls
                    fig3.add_trace(
                        go.Scatter(
                            x=df[mask][col1], y=df[mask][col2], mode='markers',
                            marker=dict(size=4, color=color_map[cls]),
                            showlegend=(i==0 and j==1), # Show legend only for the first subplot
                            name=str(cls)
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
                       title_text="Pairplot of Selected Variables with Classes",
                       legend_title=class_column
                       )
    
    # Show plot
    return fig3

# joint-plot
def update_joint_plot(selected_numeric_variable, target_variable):
    # Generate a joint plot
    fig = px.scatter(dataframe, x=selected_numeric_variable, y=target_variable, 
                    marginal_x='histogram', marginal_y='histogram')

    # Update layout for better visualization
    fig.update_layout(title=f'Joint Plot of {selected_numeric_variable} vs {target_variable}',
                    xaxis_title=selected_numeric_variable,
                    yaxis_title=target_variable)

    return fig


# RELATIONS BETWEEN NOMINALS
# mutual-information-heatmap
def update_mutual_information_heatmap(mutual_information_choice):
    """ Plots the mutual information. If only_nominals = True, it only displays the heatmap of the nominal features. If false, it shows the
    mutual information heatmap of all of the features
    """
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
        # Initialize the mutual information matrix
        mi_matrix = pd.DataFrame(index=dataframe.columns, columns=dataframe.columns, dtype=float)

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
                    mi = mutual_info_classif(dataframe[[col1]], dataframe[col2])
                    mi_matrix.loc[col1, col2] = mi[0]

        # Calculate mutual information between numeric and nominal features
        kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        for col1 in numeric_features:
            for col2 in nominal_features:
                mi = mutual_info_classif(kbin.fit_transform(dataframe[[col1]]), dataframe[col2])
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

# entropy-plot

def update_entropy_plot():
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




# mosaic-plot
# Function to create a mosaic plot
def update_mosaic_plot(variable1, variable2):
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






# RELATIONS BETWEEN NUMERIC AND NOMINAL
# categorical_boxplot
def update_categorical_boxplots(nominal_feature, numeric_feature, boxplot_order="numeric"):
    # Create horizontal boxplot traces for each category in the nominal feature
    fig = go.Figure()
    
    if boxplot_order == "numeric":
        for category in dataframe[nominal_feature].unique():
            fig.add_trace(go.Box(
                x=dataframe[dataframe[nominal_feature] == category][numeric_feature],
                y=[category] * len(dataframe[dataframe[nominal_feature] == category]),
                name=category,
                orientation='h',
                boxmean=True
            ))

        # Update layout
        fig.update_layout(
            title=f"Boxplot of {numeric_feature} by {nominal_feature}",
            xaxis_title=numeric_feature,
            yaxis_title=nominal_feature,
            showlegend=False,
            boxmode='group'
        )
        
    if boxplot_order == "category_size":
        category_counts = dataframe[nominal_feature].value_counts()

        # Sort the categories by their counts
        sorted_categories = list(category_counts.sort_values(ascending=True).index)

        for category in sorted_categories:
            count = category_counts[category]
            fig.add_trace(go.Box(
                x=dataframe[dataframe[nominal_feature] == category][numeric_feature],
                y=[category] * len(dataframe[dataframe[nominal_feature] == category]),
                name=f"{category} (n={count})",
                orientation='h',
                boxmean=True
            ))

        # Update layout
        fig.update_layout(
            title=f"Boxplot of {numeric_feature} by {nominal_feature}",
            xaxis_title=numeric_feature,
            yaxis_title=nominal_feature,
            yaxis=dict(
                tickmode='array',
                tickvals=sorted_categories,
                ticktext=[f"{category} (n={category_counts[category]})" for category in sorted_categories]
            ),
            # legend=dict(
            #     title=nominal_feature
            # ),
            showlegend=False,
            boxmode='group'
        )
        
    return fig



# swarmplot
def update_swarmplot(nominal_feature, numeric_feature):
    # Create horizontal boxplot traces for each category in the nominal feature
    # Create the figure
    fig = go.Figure()

    # Add scatter plots for each category in the nominal feature
    for category in dataframe[nominal_feature].unique():
        subset = dataframe[dataframe[nominal_feature] == category]
        fig.add_trace(go.Scatter(
            x=subset[numeric_feature],
            y=jitter(np.full_like(subset[numeric_feature], category)),
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
            tickvals=dataframe[nominal_feature].unique(),
            ticktext=dataframe[nominal_feature].unique()
        ),
        showlegend=False
    )

    return fig




