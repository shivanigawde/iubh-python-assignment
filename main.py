import math
from function import FunctionManager
from regression import minimise_loss, find_classification
from lossfunction import squared_error
from plotting import plot_ideal_functions, plot_points_with_their_ideal_function
from utils import write_deviation_results_to_sqlite

# This is the key factor of given rules. Given in assignment
ACCEPTED_FACTOR = math.sqrt(2)

if __name__ == '__main__':
    # Provide paths for csv files
    ideal_path = "data/ideal.csv"
    train_path = "data/train.csv"

    # FunctionManager takeup a path to csv & parses Function objects from the data.
    # A Function  stores 'A' and 'B' point, also it prefer Pandas to perform composedly
    candidate_ideal_function_manager = FunctionManager(path_of_csv=ideal_path)
    train_function_manager = FunctionManager(path_of_csv=train_path)

    # We will add the  suffix for complying as per the given requirement of structure's table
    # FunctionManager deploy .to_sql function from Pandas
    train_function_manager.to_sql(file_name="training", suffix=" (training func)")
    candidate_ideal_function_manager.to_sql(file_name="ideal", suffix=" (ideal func)")

    # We can store atleast 50 function in ideal_function_manager. 
    # We can store four function in train_function_manager.
    # furthermore we can use this data to compute an IdealFunction.
    # IdealFunction beyond this stores best fitting function, train data can compute the tolerance.
    # Now we required iteration over all train_functions
    # The corresponding ideal functions are stored in a list.
    ideal_functions = []
    for train_function in train_function_manager:
        # minimise_loss compute the best fitting function stated for the train function
        ideal_function = minimise_loss(training_function=train_function,
                                       list_of_candidate_functions=candidate_ideal_function_manager.functions,
                                       loss_function=squared_error)
        ideal_function.tolerance_factor = ACCEPTED_FACTOR
        ideal_functions.append(ideal_function)

    # We can  do categorization for highlighting some plotting
    plot_ideal_functions(ideal_functions, "train_and_ideal")

    # FunctionManager provides all  basic requirment to load a CSV, so we will reused it.
    # Instead of multiple Functions like before, it will now contain a single "Function" at location [0]
    # Benefit is that we can recapitulate over each point with the Function object
    test_path = "data/test.csv"
    test_function_manager = FunctionManager(path_of_csv=test_path)
    test_function = test_function_manager.functions[0]

    points_with_ideal_function = []
    for point in test_function:
        ideal_function, delta_y = find_classification(point=point, ideal_functions=ideal_functions)
        result = {"point": point, "classification": ideal_function, "delta_y": delta_y}
        points_with_ideal_function.append(result)

    # Recap: We can store list of dictionaries in points_with_ideal_functions.
    # These dictionaries depict the classification result of each point.

    # We can plot all the points with the commensurate classification function
    plot_points_with_their_ideal_function(points_with_ideal_function, "point_and_ideal")

    # At last the dict object used to write it to a sqlite
    # In this method a pure SQLAlchamy approach has been choosen with a MetaData object to save us from SQL-Language
    write_deviation_results_to_sqlite(points_with_ideal_function)
    print("following files created:")
    print("training.db: All training functions as sqlite database")
    print("ideal.db: All ideal functions as sqlite database")
    print("mapping.db: output of point-test in which the ideal function and its delta is computed")
    print("train_and_ideal.html: View the train data as distributed and the best fitting ideal function as curve")
    print("points_and_ideal.html: View for those point matching ideal function the distance between them in a figure")

    print("Initiator: Shivani Gawde")
    print("Script run successfully")
    print("Date: 22. May 2022")


