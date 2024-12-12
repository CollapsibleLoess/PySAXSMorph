from scipy import stats


def linear_fitting(calculated_data, x_key, y_key, start_index, end_index):
    x_values = calculated_data[x_key]
    y_values = calculated_data[y_key]

    x_length = len(x_values)

    int_start_index = int(x_length * start_index)
    int_end_index = int(x_length * end_index)

    x_subset = x_values[int_start_index:int_end_index]
    y_subset = y_values[int_start_index:int_end_index]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_subset, y_subset)

    return slope, intercept, r_value

