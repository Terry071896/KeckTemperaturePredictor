# Author: Terry Cox
# GitHub: https://github.com/Terry071896/KeckTemperaturePredictor
# Email: tcox@keck.hawaii.edu, tfcox1703@gmail.com

__author__ = ['Terry Cox', 'Shui Kwok']
__version__ = '1.0.1'
__email__ = ['tcox@keck.hawaii.edu', 'tfcox1703@gmail.com', 'skwok@keck.hawaii.edu']
__github__ = 'https://github.com/Terry071896/KeckTemperaturePredictor'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from progressbar import ProgressBar
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
from createTrainTest import CreateTrainTest
import multiprocessing as mp
import csaps

class TemperatureModels(object):
    '''
    A class used to load, predict, and update the trained time-series linear models to predict the temperature on top of Mauna Kea between 5-11pm at night from 4-8 hours before hand.

    ...

    Attributes
    ----------
    models : dict
        a dictionary of each of the models to predict at each hour between 4-8 hours ahead.

    Methods
    -------
    predict(x, utc_date)
        this method will choose and predict based off of the observation given 'x' and utc timestames 'utc_dates'.
    predict_all(x, utc_date)
        this method will choose all the models that it possibly can to predict based off of the observation given 'x' and utc timestames 'utc_dates' (if the length is greater than required for some models, it will take the correct amount to predict.)
    check_x(x, utc_dates)
        this method will make sure to correctly interpolate/smooth the temperature data based off of the 'utc_dates' timestamps.
    update(hours_ahead, filename, fraction_data = 1)
        this method updates the correct model, based on more/new data from 'filename'.
    _update(x)
        this method is used to access method 'update' for multiprocessing purposes.
    update_all(filename, fraction_data = 1)
        this method will update all the models (models predicting between 4 and 8 hours ahead).
    save()
        this method will write the models to a txt file.
    '''

    def __init__(self):
        self.models = {'4h_coef_linear_model':None,
            '5h_coef_linear_model':None,
            '6h_coef_linear_model':None,
            '7h_coef_linear_model':None,
            '8h_coef_linear_model':None}

        for model_name in self.models.keys():
            json_file = open(model_name+'.txt')
            json_str = json_file.read()
            self.models[model_name] = json.loads(json_str)

    def check_x(self, x, utc_dates):
        '''this method will make sure to correctly interpolate/smooth the temperature data based off of the 'utc_dates' timestamps.

        Parameters
        ----------
        x : list
            this should be a list of all the temperature values used as predictors.  Each value should be 6 minutes appart and should be 160, 170, 180, 190, or 200 values in length.
        utc_dates : list
            this should be a list of the utc timestamp of the temperature value.

        Returns
        -------
        list, list
            Each list should be the lists are the same as the parameters but adjusted to have 6 minute intervals.
        '''
        if not isinstance(x, list) or not isinstance(utc_dates, list):
            flag = True
            try:
                doesItWork = x[0]
            except:
                print('\'x\' must be a list.')
                flag = False
            try:
                doesItWork = utc_dates[0]
            except:
                print('\'utc_dates\' must be a list.')
                flag = False
            if not flag:
                return None, None


        if len(x) != len(utc_dates):
            print('\'x\' and \'utc_dates\' not the same length: %s and %s'%(len(x), len(utc_dates)))
            return None, None

        if not isinstance(utc_dates[0], datetime):
            print('\'utc_dates\' needs to be a list of datetime values. Currently, %s'%(type(utc_dates[0])))
            return None, None

        if not isinstance(x[0], int) or not isinstance(x[0], float):
            print('\'x\' needs to be a list of int or float values. Currently, %s'%(type(x[0])))
            return None, None

        dt = np.diff(utc_dates)
        t = []
        s = 0
        for i in dt:
            t.append(s)
            s += i.minutes + i.seconds/60
        t.append(s)

        sp0 = csaps.UnivariateCubicSmoothingSpline(t, x, smooth=0.01)
        xs0 = np.array(range(int(np.ceil(t[-1]/6))+1))*6
        x_new = sp0(xs0)

        new_utc_dates = []
        for six in xs0:
            temp_date = utc_dates[0] + timedelta(minutes=int(six))
            new_utc_dates.append(temp_date)

        return x_new, new_utc_dates




    def predict_all(self, x, utc_dates):
        '''this method will choose and predict based off of the observation given 'x' and utc timestames 'utc_dates'.
        If the length is greater than required for some models, it will take the correct amount to predict.
        For example, if the length of 'x' is 180, then this method will predict for 6, 7, and 8 hours ahead as there is enough information to do so.

        Parameters
        ----------
        x : list
            this should be a list of all the temperature values used as predictors.  Each value should be 6 minutes appart and should be 160, 170, 180, 190, or 200 values in length.
        utc_dates : list
            this should be a list of the utc timestamp of the temperature value.

        Returns
        -------
        dict
            A dictionary of dictionarys containing predicted values as well as 68% and 95% confidence intervals of each of the possible predicted models.
            An empty dictionary will be returned if the time is outside the predictive window.
        '''

        pred_models = {}
        x, utc_dates = self.check_x(x, utc_dates)

        if x is None or utc_dates is None:
            print('Input for \'x\' should be list of int or floats.')
            print('Input for \'utc_dates\' should be a list of datetime values.')
            print('One or both are wrong')
            return pred_models

        utc_date = utc_dates[-1]


        if 3 < utc_date.hour < 18:
            return pred_models

        for model_name in self.models.keys():
            n_temp_predictors = len(self.models[model_name]['coef'])-4
            hours_ahead = int(model_name[0])
            utc_hours_ahead = (utc_date.hour+hours_ahead)%23

            if len(x) >= n_temp_predictors and 3 <= utc_hours_ahead <= 7:
                x_temp = x[-n_temp_predictors:]
                pred_models[model_name] = self.predict(x_temp, utc_date, hours_ahead)
            else:
                pred_models[model_name] = {}

        return pred_models


    def predict(self, x, utc_dates, hours_ahead):
        '''this method will choose and predict based off of the observation given 'x' and utc timestames 'utc_dates'.

        Parameters
        ----------
        x : list
            this should be a list of all the temperature values used as predictors.  Each value should be 6 minutes appart and should be 160, 170, 180, 190, or 200 values in length.
        utc_date : list
            this should be a list of utc timestamps of the temperature value.
        hours_ahead : int
            this should be the number of hours ahead you are trying to predict. Only ints between 4-8 are acceptable.

        Returns
        -------
        dict
            A dictionary of predicted values as well as 68% and 95% confidence intervals.
            An empty dictionary will be returned if the time is outside the predictive window.
        '''

        pred_model = {'pred_val':None,
            '68_conf_interval':None,
            '95_conf_interval':None}

        if hours_ahead not in [4, 5, 6, 7, 8]:
            print('\'hours_ahead\' needs to be an int 4-8 only.')
            return pred_model

        if isinstance(utc_dates, datetime):
            utc_date = utc_dates
        else:
            x, utc_dates = self.check_x(x, utc_dates)
            if x is None or utc_dates is None:
                print('Input for \'x\' should be list of int or floats.')
                print('Input for \'utc_dates\' should be a list of datetime values.')
                print('One or both are wrong')
                return pred_model
            else:
                utc_date = utc_dates[-1]


        utc_hours_ahead = (utc_date.hour+hours_ahead)%23

        model_name = '%sh_coef_linear_model'%(int(hours_ahead))

        if (240-hours_ahead*10) <= len(x):
            n_time_predictors = 240-hours_ahead*10
        else:
            print('Size of \'x\' is not big enough for prediction: %s'%(len(x)))
            return pred_model

        if len(x) == n_time_predictors and 3 <= utc_hours_ahead <= 7:
            x = np.append(x, utc_date.month)
            x = np.append(x, utc_date.day)
            x = np.append(x, utc_hours_ahead)
            x = np.append(x, utc_date.minute)

            pred_val = np.dot(self.models[model_name]['coef'], x)+self.models[model_name]['intercept']
            upper68 = pred_val + self.models[model_name]['mean'] + self.models[model_name]['std']
            lower68 = pred_val + self.models[model_name]['mean'] - self.models[model_name]['std']
            upper95 = pred_val + self.models[model_name]['mean'] + 2*self.models[model_name]['std']
            lower95 = pred_val + self.models[model_name]['mean'] - 2*self.models[model_name]['std']

            pred_model['pred_val'] = pred_val
            pred_model['68_conf_interval'] = [lower68, upper68]
            pred_model['95_conf_interval'] = [lower95, upper95]

        return pred_model

    def update(self, hours_ahead, filename, fraction_data = 1):
        '''this method updates the correct model, based on more/new data from 'filename'.

        Parameters
        ----------
        hours_ahead : int
            the value of how many hours ahead we are trying to predict.  The value must be an int between 4 and 8 (including both 4 and 8).
        filename : str
            the name of the file name that contains new data to update the models.  Must have columns: `Date`, `UT`, and `OutTemp`
        test_size : float, optional (default = 0.1)
            the value of what percentage of X and y should be the saved for the testing sets.  The value should be between 0 and 1 (not including 0).

        Returns
        -------
        dict
            the dictionary of dictionaries that contains the newly trained models.
        '''

        TnT = CreateTrainTest(filename, fraction_data)
        TnT.create(hours_ahead)

        model_name = '%sh_coef_linear_model'%(hours_ahead)
        model = LinearRegression().fit(TnT.X_train, TnT.y_train)
        model_full = LinearRegression().fit(TnT.X, TnT.y)

        y_pred = model.predict(TnT.X_test)
        Error = (np.array(TnT.y_test)-np.array(y_pred))

        self.models[model_name]['coef'] = list(model_full.coef_)
        self.models[model_name]['intercept'] = model_full.intercept_.tolist()
        self.models[model_name]['mean'] = np.mean(Error)
        self.models[model_name]['std'] = np.std(Error)

        return self.models[model_name]

    def _update(self, x):
        '''this method is used to access method 'update' for multiprocessing purposes.

        Parameters
        ----------
        x : tuple, list
            this tuple/list should contain the necessay parameters (in order) for the 'update' method.

        Returns
        -------
        dict
            the dictionary of dictionaries that contains the newly trained models.
        '''

        return self.update(x[0], x[1], x[2])

    def update_all(self, filename, fraction_data = 1):
        '''this method will update all the models (models predicting between 4 and 8 hours ahead).

        Parameters
        ----------
        filename : str
            the name of the file name that contains new data to update the models.  Must have columns: `Date`, `UT`, and `OutTemp`
        fraction_data : float, optional (defalut = 1)
            the percentage of the data that you want to clean and parse into X and y values.  This value should be in the range of 0 to 1 (not including 0).
        '''

        pool = mp.Pool(mp.cpu_count()) # Start number of threads
        updated_models = pool.map(self._update, [(int(model_name[0]), filename, fraction_data) for model_name in self.models.keys()]) #
        pool.close() # Kill threads

        #self.save()
        return updated_models

    def save(self):
        '''this method will write the models to a txt file.'''

        for model_name in self.models.keys():
            json_txt = json.dumps(self.models[model_name], indent=4)
            with open(model_name+'.txt', 'w') as file:
                file.write(json_txt)
            print('Saving %s'%(model_name))
