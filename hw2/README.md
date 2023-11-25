# Homework 2: Gradient Methods for Optimization

This project contains the implementation and evaluation of various gradient methods for optimization. The focus is on applying these methods to the Iris dataset for classification tasks.

## Project Structure

- `iris/`: Directory containing the Iris dataset used for the optimization tasks.
- `plot/`: Directory for storing generated plots and figures from the analysis.
- `methods.py`: Contains the implementation of different gradient methods like sub-gradient, proximal gradient, and accelerated proximal gradient methods.
- `sanityCheck.py`: Script to perform sanity checks on the methods implemented. Comparison with sklearn library.
- `q4.py`: Script that executes the gradient methods on the Iris dataset and that creates the plot. (question 4 of the homework)
- `q2.py`: Script that executes the gradient and the derivative free methods and plot the results 
- `makefile`: A makefile to automate the execution of scripts.

## Packages Used

To run the project you need these packages to be installed:

- `pandas`: For reading and manipulating the Iris dataset.
- `numpy`
- `matplotlib`
- `scikit-learn`: Only used for the sanity check. 


## Running the Project

To run the main script, you can use the makefile provided:

```bash
make run
