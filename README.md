#Titanic Survivor Analysis
 This project is a Titanic Survivor Analysis, a very popular [Kaggle 
 competition](https://www.kaggle.com/c/titanic) of 
 predicting the survival of Titanic passengers.
 
 The first part of the project contains missing values imputation, 
 in the following R scripts:
 * ageImputation.R
 * embarkedImputation.R
 * fareImputation.R  

 Available reports _report1.csv_ and _report2.csv_ show respectively:
 * Survivors by class, age band and sex
 * Count of survivors by state
 
 Predictive modeling is implemented using Spark 2.1 in Databricks 
 Community Edition, since the dataset is small in size. Provided 
  notebook (_titanic-analysis.html_) contains exploration, feature
  engineering, predictive modeling and dimensionality reduction,
   with provided comments and notes. PySpark script is also 
   available as _titanic-analysis.py_ PySpark script.
   
   © Milos Milovanovic, Things Solver
 