from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, grid
from bokeh.models import Band, ColumnDataSource


def plot_ideal_functions(ideal_functions, file_name):
    """
    Plots all ideal functions
    :param ideal_functions: list of ideal functions
    :param file_name: the name the .html file should get
    """
    ideal_functions.sort(key=lambda ideal_function: ideal_function.training_function.name, reverse=False)
    plots = []
    for ideal_function in ideal_functions:
        p = plot_graph_from_two_functions(line_function=ideal_function, scatter_function=ideal_function.training_function,
                                          squared_error=ideal_function.error)
        plots.append(p)
    output_file("{}.html".format(file_name))
    # how unpacking is used to perform the arguments
    show(column(*plots))


def plot_points_with_their_ideal_function(points_with_classification, file_name):
    """
    Plot all points that have a matched classification
    :param points_with_classification: a list containing dicts with "classification" and "point"
    :param file_name: the name the .html file should get
    """
    plots = []
    for index, item in enumerate(points_with_classification):
        if item["classification"] is not None:
            p = plot_classification(item["point"], item["classification"])
            plots.append(p)
    output_file("{}.html".format(file_name))
    show(column(*plots))


def plot_graph_from_two_functions(scatter_function, line_function, squared_error):
    """
    plots a scatter for the train_function and a line for the ideal_function
    :param scatter_function: the train function
    :param line_function: ideal function
    :param squared_error: the squared error will be plotted in the title
    """
    f1_dataframe = scatter_function.dataframe
    f1_name = scatter_function.name

    f2_dataframe = line_function.dataframe
    f2_name = line_function.name

    squared_error = round(squared_error, 2)
    p = figure(title="train model {} vs ideal {}. Total squared error = {}".format(f1_name, f2_name, squared_error),
               x_axis_label='x', y_axis_label='y')
    p.scatter(f1_dataframe["x"], f1_dataframe["y"], fill_color="red", legend_label="Train")
    p.line(f2_dataframe["x"], f2_dataframe["y"], legend_label="Ideal", line_width=2)
    return p


def plot_classification(point, ideal_function):
    """
    plots the classification function and a point on top. It also displays the tolerance
    :param point: a dict with "x" and "y"
    :param ideal_function: a classification object
    """
    if ideal_function is not None:
        classification_function_dataframe = ideal_function.dataframe

        point_str = "({},{})".format(point["x"], round(point["y"], 2))
        title = "point {} with classification: {}".format(point_str, ideal_function.name)

        p = figure(title=title, x_axis_label='x', y_axis_label='y')

        # Create the ideal function
        p.line(classification_function_dataframe["x"], classification_function_dataframe["y"],
                legend_label="Classification function", line_width=2, line_color='black')

        # Steps to display the tolerance within the graph
        criterion = ideal_function.tolerance
        classification_function_dataframe['upper'] = classification_function_dataframe['y'] + criterion
        classification_function_dataframe['lower'] = classification_function_dataframe['y'] - criterion

        source = ColumnDataSource(classification_function_dataframe.reset_index())

        band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay',
            fill_alpha=0.3, line_width=1, line_color='green', fill_color="green")

        p.add_layout(band)

        # draw the point
        p.scatter([point["x"]], [round(point["y"], 4)], fill_color="red", legend_label="Test point", size=8)

        return p
