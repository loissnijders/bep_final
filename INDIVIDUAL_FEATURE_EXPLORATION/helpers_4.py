import sys
import os
# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from load_data import dataframe, metadata, target_feature, type_target_feature

nominal_features = list(metadata[metadata["DataType"] == "nominal"].Attribute)
numeric_features = list(metadata[metadata["DataType"] == "numeric"].Attribute)

target_feature = metadata[metadata.Target=='true'].Attribute.iloc[0]
type_target_feature = metadata[metadata.Target=='true'].DataType.iloc[0]