# Solving Regression and Classification Problems using Python

## Introduction
This project is a program that uses training data to choose four best functions, which are a best fit, out of 50. The program uses the test data provided to determine, for each x-y pair values, whether or not they can be assigned to the four best functions. The program also includes some logical data visualization.  

## Files/Modules
The program contains several modules for easier understanding and readability. Here's what each module does or contains:

- `exceptions.py` contains custom exceptions created for use within the program
- `function.py` contains Function classes that help manipulate the x & y values of a Function and also to handle prediction Function, regression and training data
- `iterators.py` contains classes that help loop over Manager and Function instances
- `manager.py` contains the Manager class that helps with excel file parsing and writing data to an SQL database
- `regression.py` contains functions which return the Best Function based on the training function and determine whether or not a point is within the tolerance of a classification
- `utils.py` contains utility functions that plot all Best Functions, points that match a classification and a classification function with a point on top 
- `main.py` is the program's entry point which makes use of various classes and functions from other modules to perform necessary calculations, plottings and data manipulations

## How To Run
To run this project, take the following steps:
- Extract the `functions` directory from the zipped folder
- Navigate into the `functions` directory
```
$ cd functions
```
- Create a virtual environment
```
$ python3 -m venv functionsenv
```
- Activate the virtual environment
```
$ source ./functionsenv/bin/activate
```
- Install the requirements
```
$ pip install -r requirements.txt
```
- Run the program
```
$ python3 main.py
```
