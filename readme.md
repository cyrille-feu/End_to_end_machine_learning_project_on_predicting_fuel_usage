## Predicting Fuel usage : end to end machine learning

The goal of this project is to develop an End-to-End Machine Learning Project and Deploy it to Heroku with Flask
we will go through all the major steps involved in completing an and-to-end Machine Learning project. to make this simple, we are going to choose a supervised learning regression problem.

#### Data
We are going to use the Auto MPG dataset from the UCI Machine Learning Repository. Here is the link to the dataset: http://archive.ics.uci.edu/ml/datasets/Auto+MPG

The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes

        - Attribute Information:

            1. mpg: continuous
            2. cylinders: multi-valued discrete
            3. displacement: continuous
            4. horsepower: continuous
            5. weight: continuous
            6. acceleration: continuous
            7. model year: multi-valued discrete
            8. origin: multi-valued discrete
            9. car name: string (unique for each instance)



#### Problem Statement
The data contains the variable MPG which is a continuous data and tells us about the fuel efficiency of a vehicle in the 1970s and 1980s.
Our objective here is to predict the MPG value of a vehicle given that we have other attributes of that vehicle.

#### Steps
- EDA : Carry our exploratory analysis to figure out the important features and creating new combination of features.
- Data Preparation : create a pipeline of tasks to transform the data to be loaded into our ML models.
- Selecting and Training ML models : Training a few models to evaluate their predictions using cross-validation.
- Hyperparameter Tuning : Fine tune the hyperparameters for the models that showed promising results.
- Deploy the Model using a web service : Using Flask web framework to deploy our trained model on Heroku
