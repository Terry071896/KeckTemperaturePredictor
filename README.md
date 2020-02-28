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

### Class : TemperatureModels

A class used to load, predict, and update the trained time-series linear models to predict the temperature on top of Mauna Kea between 5-11pm at night from 4-8 hours before hand.

  ...

  Attributes
  ----------
  models : dict
      a dictionary of each of the models to predict at each hour between 4-8 hours ahead.

  Methods
  -------
  predict(x, utc_date = None)
      this method will choose and predict based off of the observation given 'x'.
  predict_all(x, utc_date = None)
      this method will choose all the models that it possibly can to predict based off of the observation given 'x' (if the length is greater than required for some models, it will take the correct amount to predict.)
  update(hours_ahead, filename, fraction_data = 1)
      this method updates the correct model, based on more/new data from 'filename'.
  _update(x)
      this method is used to access method 'update' for multiprocessing purposes.
  update_all(filename, fraction_data = 1)
      this method will update all the models (models predicting between 4 and 8 hours ahead).
  save()
      this method will write the models to a txt file.
