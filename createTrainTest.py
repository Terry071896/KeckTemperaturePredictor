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

class CreateTrainTest(object):

    def __init__(self, file = None):
        if file == None:
            self.file = 'k1_temp_mirror_5min.csv'
        else:
            self.file = file

        dataSet = pd.read_csv(self.file)

        for i in range(len(dataSet[' OutTemp'])):
            vals = [i-2,i-1,i+1,i+2]
            x = np.array(vals)-i
            if -17 > dataSet[' OutTemp'][i] or dataSet[' OutTemp'][i] > 17:
                y = dataSet[' OutTemp'][vals]
                if -17 < np.min(y) and np.max(y) < 17:
                    tck = interpolate.splrep(x, y, s=0)
                    new_x = interpolate.splev(0, tck, der=0)
                    dataSet.at[i, ' OutTemp'] = new_x

        dataSet = dataSet.drop(dataSet[dataSet[' OutTemp'] < -17].index).reset_index(drop=True)
        dataSet = dataSet.drop(dataSet[dataSet[' OutTemp'] > 17].index).reset_index(drop=True)

        format = '%Y-%m-%d %H:%M'
        dataSet['Datetime'] = pd.to_datetime(dataSet['Date'] + dataSet[' UT'], format=format)
        dataSet['year'] = pd.DatetimeIndex(dataSet['Datetime']).year
        dataSet['month'] = pd.DatetimeIndex(dataSet['Datetime']).month
        dataSet['day'] = pd.DatetimeIndex(dataSet['Datetime']).day
        dataSet['hour'] = pd.DatetimeIndex(dataSet['Datetime']).hour
        dataSet['minute'] = pd.DatetimeIndex(dataSet['Datetime']).minute

        dataSet.columns = dataSet.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        self.df = dataSet

    def create(self, hours_ahead):
        self.n_predictors = (24-hours_ahead)*10
        self.hours_ahead = hours_ahead

        data = self.df['OutTemp']
        time = self.df['UT']
        date = self.df['Date']
        length = len(data)
        X = []
        y = []
        y_date = []
        pbar = ProgressBar()
        for i in pbar(range(0,length-n_predictors-hours_ahead*10)):
            x = []
            t = []
            d = []
            labels = []
            try:
                if 3 <= self.df.hour[i+n_predictors+hours_ahead*10-1] <= 7:
                    temp_hour = dataSet.hour[i+n_predictors+hours_ahead*10-1]
                else:
                    raise ValueError('not right range')

                for j in range(0, n_predictors):
                    string = date[i+j]+' '+time[i+j]
                    date1 = datetime.strptime(string, '%Y-%m-%d %H:%M')
                    d.append(date1)

                    x.append(data[i+j])
                    labels.append('%s_hr_ago'%((j+1)*6/60))

                import csaps
                t = np.array(range(len(x)))*6
                sp0 = csaps.UnivariateCubicSmoothingSpline(t, x, smooth=0.01)
                xs0 = np.linspace(t[0], t[-1], n_predictors)
                x = sp0(xs0)


                y_temp = data[i+n_predictors+hours_ahead-1]

                date_x_prob = np.sum(np.diff(d) != timedelta(minutes=6))
                string = date[i+n_predictors+hours_ahead-1]+' '+time[i+n_predictors+hours_ahead-1]
                date_y = datetime.strptime(string, '%Y-%m-%d %H:%M')
                date_y_prob = np.sum(np.diff([d[-1], date_y]) != timedelta(minutes=hours_ahead*6))

                x = np.append(x, self.df.month[i+n_predictors+hours_ahead-1])
                x = np.append(x, self.df.day[i+n_predictors+hours_ahead-1])
                x = np.append(x, self.df.hour[i+n_predictors+hours_ahead-1])
                x = np.append(x, self.df.minute[i+n_predictors+hours_ahead-1])
                labels.append('Month')
                labels.append('Day')
                labels.append('Hour')
                labels.append('Minute')
                if date_x_prob + date_y_prob == 0:
                    X.append(x)
                    y.append(y_temp)
                    y_date.append(date_y)

            except:
                continue

        self.X = np.array(X)
        self.y = np.array(y)
        self.y_date = np.array(y_date)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.labels = np.array(labels)
