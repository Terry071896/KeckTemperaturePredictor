# KeckTemperaturePredictor

## Introduction
This repository contains the necessary files to load, predict, and update the trained time-series linear models to predict the temperature on top of Mauna Kea between 5-11pm at night from 4-8 hours before hand.

## Install
Runs python3

```
git clone https://github.com/Terry071896/KeckTemperaturePredictor
cd .../KeckTemperaturePredictor
sudo pip install -r requirements.txt
```

##

***

# Class : TemperatureModels

A class used to load, predict, and update the trained time-series linear models to predict the temperature on top of Mauna Kea between 5-11pm at night from 4-8 hours before hand.


  Attributes
  ----------
  - models : dict

      a dictionary of each of the models to predict at each hour between 4-8 hours ahead.

  Methods
  -------
  - predict(x, utc_date = None)

      this method will choose and predict based off of the observation given 'x'.

  - predict_all(x, utc_date = None)

      this method will choose all the models that it possibly can to predict based off of the observation given 'x' (if the length is greater than required for some models, it will take the correct amount to predict.)

  - update(hours_ahead, filename, fraction_data = 1)

      this method updates the correct model, based on more/new data from 'filename'.

  - _update(x)

      this method is used to access method 'update' for multiprocessing purposes.

  - update_all(filename, fraction_data = 1)

      this method will update all the models (models predicting between 4 and 8 hours ahead).

  - save()

      this method will write the models to a txt file.

***

# Class : CreateTrainTest

A class used to clean and parse data into workable machine learning time-series data from a `csv` file that has recorded temperatures of from the Keck I temperature sensor.
  The format of the data should have a columns: `Date`, `UT`, and `OutTemp`.


  Attributes
  ----------
  - file : str

      the file name of the csv file contained the temperature readings and times from the Keck I sensor.

  - df : dataframe

      the data loaded into a pandas dataframe of the temperature readings and times from the Keck I sensor (from `filename` csv)

  - n_temp_predictors : int

      the number of predictors that are temperature related

  - n_predictors : int

      the total number of predictors

  - hours_ahead : int

      the hours ahead that we are trying to predict

  - X : matrix (numpy array)

      the matrix of observations (each row is a observation; each column is a predictor).  This is used to train time-series model.

  - y : numpy array

      the list of true values to predict of each given observation (row) of matrix X.

  - y_date : numpy array

      the list of timestamps associated with the temperatures in list y.

  - X_train : matrix (numpy array)

      the subset of matrix X that are used to train the time-series model.

  - X_test : matrix (numpy array)

      the remaining subset of matrix X (observations not in X_train) that are used to test the trained model on X_train.

  - y_train : numpy array

      the list of true values to predict of each given observation (row) of matrix X_train.

  - y_test : numpy array

      the list of true values to predict of each given observation (row) of matrix X_test

  - labels : numpy array

      the list the variables labeles.

  Methods
  -------
  - create(self, hours_ahead, test_size = 0.1)

      parses, cleans, and adds good observations to X and y.  Then, finishes by splitting into training and testing sets.
