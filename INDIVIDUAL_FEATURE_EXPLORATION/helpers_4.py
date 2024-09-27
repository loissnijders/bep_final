import numpy as np
from dash import dcc, html

def calculate_bins_freedman_diaconis(data):
    data = data.dropna()  # Remove NaN values
    n = data.size
    if n == 0:
        return 1  # Avoid division by zero if data is empty

    # Calculate IQR
    q75, q25 = np.percentile(data, [75, 25])
    IQR = q75 - q25
    if IQR == 0:
        IQR = data.std()  # Use standard deviation if IQR is zero
        if IQR == 0:
            return 1  # All data points are identical

    # Calculate bin width
    bin_width = (2 * IQR) / (n ** (1/3))
    if bin_width == 0:
        bin_width = 1  # Set bin width to 1 if calculation results in zero

    # Calculate number of bins
    data_range = data.max() - data.min()
    k = int(np.ceil(data_range / bin_width))
    k = max(1, k)  # Ensure at least one bin

    return k

def define_slider_range(k_recommended, n_unique):
    # Define delta as 50% of the recommended bins or at least 5
    delta = max(5, int(0.5 * k_recommended))
    
    # Calculate slider min and max
    k_min = max(1, k_recommended - delta)
    k_max = min(k_recommended + delta, n_unique, 100)  # Cap at 100 bins

    # Ensure k_min does not exceed k_max
    if k_min > k_max:
        k_min = max(1, k_max - 5)

    return k_min, k_max

def get_bin_slider(dataframe, feature):
    data = dataframe[feature].dropna()
    n_unique = data.nunique()

    # Calculate the recommended number of bins
    k_recommended = calculate_bins_freedman_diaconis(data)

    # Define slider range
    k_min, k_max = define_slider_range(k_recommended, n_unique)

    # Generate slider marks
    step_size = max(1, (k_max - k_min) // 5)
    marks = {i: str(i) for i in range(k_min, k_max + 1, step_size)}

    # Create the slider
    slider = dcc.Slider(
        id={'type': 'bin-slider', 'index': feature},
        min=k_min,
        max=k_max,
        step=1,
        value=k_recommended,
        marks=marks
    )
    
    return slider


explanation_number_bins = "The optimal number of bins for the histogram is calculated using the Freedman-Diaconis rule, which helps create a histogram that best represents the data distribution."