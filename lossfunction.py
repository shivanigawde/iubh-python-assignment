def squared_error(first_function, second_function):
    """
    Calculates the squared error to another function
    :param other_function:
    :return: the squared error
    """
    distances = second_function - first_function
    distances["y"] = distances["y"] ** 2
    total_deviation = sum(distances["y"])
    return total_deviation

