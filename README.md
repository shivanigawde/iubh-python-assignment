# The assignment and its algorithms
The assignment can be interpreted into two parts. In the following they are described in text. 

## Part 1: determining the best fitting ideal function
In part 1 the goal is to find for a training function the best fitting ideal function amongst 50 candidates. There are a total of 4 different training functions and for each, the ideal function has to be found. A function is a collection of x- and y- coordinates and is provided for in a .csv file. Within the 50 candidates, the ideal function which has the lowest squared error towards the training function, is defined as the ideal function. This in essence is a variation of the “Mean squared error” and is a popular loss function towards models are optimised (Kerzel, 2020). 
The Squared Error has a couple of properties 
Because the deviation is squared we always have a positive result
Large deviations have a strong impact
I now describe in written words how the algorithm which I implemented for part 1 works. The implementation itself can be found in chapter 5.
For each training function, go over each point and calculate the deviation of the y-value towards a candidate ideal function. Square the deviation and sum all up, this value is defined as Squared Error. Do this for all candidate functions and the function which has the lowest Squared Error, is the ideal function.
The result is thus 4 ideal functions.
## Part 2: classifying points
In part 2, a collection of points is provided for as .csv file, and for each point it needs to be determined if it can be assigned towards one of the 4 ideal functions. In addition, if a match is made the deviation has to be computed.
Again, in words how the assigning of points in the implementation works. Implementation details can be found in chapter 6.
For each point within the test data collection, compute the absolute linear deviation (ald) to each of the ideal functions. Determine for each deviation if it is within lower than the tolerance. If multiple fit, pick the classification with the lowest ald. The computation of the tolerance is given within the assignment, and equals to the largest deviation between training- and ideal function multiplied with sqrt(2). 
## Storing of data using SQLite
In addition to the computation, data has to be written towards a SQLite database. Three databases have to be generated.
A database which mirrors the training data set
A database which mirrors the candidate ideal functions data set
A database which stores the classification towards the ideal functions together with the deviation
Ad 3: The assignment did not provide any detail towards what should be saved if no classification can be made. Furthermore, if no classification can be made the deviation cannot be provided for either. The program writes in the case of no classification “-” within the “No of ideal func” column and “-1” within the “Delta Y (test func)” column.
## Other requirements
Within the assignment, there are further requirements that impact the design of the program significantly. Foremost, it has to demonstrate an object-oriented design and use packages such as Panda, Bokeh and SQLAlchamy. As conclusion, the point is clearly not to simply calculate and solve the assignment but to demonstrate knowledge of Python and popular packages for data science purposes.

