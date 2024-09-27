# In this file I will store functions and variables that I will use throughout

from helpers import get_data_metadata

dictionary_order_options = {
    'Categories ordered alphanumerically in ascending order': 'category ascending',
    'Categories ordered alphanumerically in descending order': 'category descending',
    'Frequencies ordered in ascending order': 'total ascending',
    'Frequencies ordered in descending order': 'total descending'
    }

dictionary_plot_options = {
    'box_plot': 'box_plot',
    'violin_plot': 'violin_plot',
    'histogram': 'histogram'
}


def feature_table_dataframes(data_id):
    """ This function creates the datatable of the contents that should go in the feature table
    Note that there can be two feature tables in one analysis app. The nominal features and numeric features go into seperate tables
    """
    df, meta_features, numerical_features, nominal_features = get_data_metadata(data_id)

    # Feature tables nominal features
    nominal_meta_features = meta_features[meta_features.DataType == 'nominal']
    nominal_meta_features = nominal_meta_features.drop(['Target'], axis=1)

    if len(nominal_features) != 0:
        for attribute in nominal_features:
            nominal_meta_features.loc[nominal_meta_features.Attribute == attribute, 'Mode'] = df[attribute].mode()[0]

    # Feature table numerical features
    numerical_meta_features = meta_features[meta_features.DataType == 'numeric']
    numerical_meta_features = numerical_meta_features.drop(['# categories', 'Target', 'Entropy'], axis=1)
    
    if len(numerical_features) != 0:
        for attribute in numerical_features:
            numerical_meta_features.loc[numerical_meta_features.Attribute == attribute, 'Mean'] = df[attribute].mean()
            numerical_meta_features.loc[numerical_meta_features.Attribute == attribute, 'Minimum'] = df[attribute].min()
            numerical_meta_features.loc[numerical_meta_features.Attribute == attribute, 'Maximum'] = df[attribute].max()
            numerical_meta_features.loc[numerical_meta_features.Attribute == attribute, 'Variance'] = df[attribute].var()
            numerical_meta_features.loc[numerical_meta_features.Attribute == attribute, 'Standard deviation'] = df[attribute].std()

    return nominal_meta_features, numerical_meta_features

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