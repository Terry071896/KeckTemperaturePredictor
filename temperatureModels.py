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
import json

class TemperatureModels(object):

    def __init__(self):
        self.models = {'4h_coef_linear_model':None,
            '5h_coef_linear_model':None,
            '6h_coef_linear_model':None,
            '7h_coef_linear_model':None,
            '8h_coef_linear_model':None}

        for filename in self.models.keys():
            json_file = open(filename+'.txt')
            json_str = json_file.read()
            self.models[filename] = json.loads(json_str)

    def predict_all(x):
        utc_date = datetime.utcnow()
        pred_models = {}
        if 3 < utc_date.hour < 18:
            return pred_models

        for filename in self.models.keys():
            n_temp_predictors = len(self.models[filename]['coef'])-4
            hours_ahead = int(filename[0])

            if len(x) >= n_temp_predictors and 3 <= utc_date.hour+hours_ahead <= 7:
                x_temp = x[-n_temp_predictors:]
                pred_models[filename] = self.predict(x_temp, utc_date)
            else:
                pred_models[filename] = {}

        return pred_val


    def predict(x, utc_date = None):
        if utc_date is None:
            utc_date = datetime.utcnow()

        pred_model = {'pred_val':None,
            '68_conf_interval':None,
            '95_conf_interval':None}

        hours_ahead = 24-len(x)/10
        filename = '%sh_coef_linear_model'%(hours_ahead)
        try:
            n_temp_predictors = len(self.models[filename]['coef'])-4
        except:
            print('Size of \'x\' is not correct for prediction: %s'%(len(x)))
            return pred_model

        if len(x) == n_temp_predictors and 3 <= utc_date.hour+hours_ahead <= 7:
            t = np.array(range(len(x)))*6
            sp0 = csaps.UnivariateCubicSmoothingSpline(t, x, smooth=0.01)
            xs0 = np.linspace(t[0], t[-1], n_time_predictors)
            x = sp0(xs0)

            x = np.append(x, utc_date.month)
            x = np.append(x, utc_date.day)
            x = np.append(x, utc_date.hour+hours_ahead)
            x = np.append(x, utc_date.minute)

            pred_val = np.dot(self.models[filename]['coef'], x)+self.models[filename]['intercept']
            upper68 = pred_val + self.models[filename]['mean'] + self.models[filename]['std']
            lower68 = pred_val + self.models[filename]['mean'] - self.models[filename]['std']
            upper95 = pred_val + self.models[filename]['mean'] + 2*self.models[filename]['std']
            lower95 = pred_val + self.models[filename]['mean'] - 2*self.models[filename]['std']

            pred_model['pred_val'] = pred_val
            pred_model['68_conf_interval'] = [lower68, upper68]
            pred_model['95_conf_interval'] = [lower95, upper95]

        return pred_model


    def save(self):
        for filename in self.models.keys():
            json_txt = json.dumps(self.models[filename], indent=4)
            with open(filename+'.txt', 'w') as file:
                file.write(json_txt)
