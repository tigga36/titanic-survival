# titanic-survival
This is a simple personal project to get acquainted with general ML and constructions of models. This README is generally intended for personal note-taking.  
  
NOTE: The data and codes handled in this project are more or less tested/analyzed in a separate Jupyter Notebook. This is git is more or less a way to summarize the progress of the project.

# The Dataset
In this project I will be using the Titanic dataset taken from Kaggle (https://www.kaggle.com/c/titanic/overview) to create a binary classifier of whether a survivor survived or not depending on various attributes.

The attributes are:  
  -PassengerId (int) ID of the boarding passenger  
  -Survived (int) Whether the passenger survived or not, with 0 dead and 1 alive  
  -Pclass (int) Class of the travel  
  -Name (str) Name of the passenger  
  -Sex (str) Gender of the passenger  
  -Age (int) Age of the passenger  
  -SibSp (str) Number of siblings/spouse on board with the passenger  
  -Parch (int) Number of parents/children on board with the passenger  
  -Ticket (str) The ticket of the passenger  
  -Fare (int) The fare paid for this passenger's trip  
  -Cabin (str) The cabin number of where the passenger stayed  
  -Embarked (str) The destination of the passenger's trip  

# Data Cleaning
The first steps of this project mainly has to do with making data better-suited to be used as training data. Examples of cleaning the data includes, are but not limited to:  
  -Converting non-numerical attributes (Sex, Name, etc.) to numerical attributes to better suite training processes  
  -Regularizing numerical attributes to prevent certain attributes from being unnecessarily prominent than others  
  -Removing attributes that does not have sufficient correlation with the survival of passengers altogether  
  -Spot anomalies in data that could potentially have considerable effect on the trained model  
