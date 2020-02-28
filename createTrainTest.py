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
import zipfile

class CreateTrainTest(object):
    '''
    A class used to clean and parse data into workable machine learning time-series data from a `csv` file that has recorded temperatures of from the Keck I temperature sensor.
    The format of the data should have a columns: `Date`, `UT`, and `OutTemp`.

    ...

    Attributes
    ----------
    file : str
        the file name of the csv file contained the temperature readings and times from the Keck I sensor.
    df : dataframe
        the data loaded into a pandas dataframe of the temperature readings and times from the Keck I sensor (from `filename` csv)
    n_temp_predictors : int
        the number of predictors that are temperature related
    n_predictors : int
        the total number of predictors
    hours_ahead : int
        the hours ahead that we are trying to predict
    X : matrix (numpy array)
        the matrix of observations (each row is a observation; each column is a predictor).  This is used to train time-series model.
    y : numpy array
        the list of true values to predict of each given observation (row) of matrix X.
    y_date : numpy array
        the list of timestamps associated with the temperatures in list y.
    X_train : matrix (numpy array)
        the subset of matrix X that are used to train the time-series model.
    X_test : matrix (numpy array)
        the remaining subset of matrix X (observations not in X_train) that are used to test the trained model on X_train.
    y_train : numpy array
        the list of true values to predict of each given observation (row) of matrix X_train.
    y_test : numpy array
        the list of true values to predict of each given observation (row) of matrix X_test
    labels : numpy array
        the list the variables labeles.

    Methods
    -------
    create(self, hours_ahead, test_size = 0.1)
        parses, cleans, and adds good observations to X and y.  Then, finishes by splitting into training and testing sets.
    '''
    def __init__(self, file = None, fraction_data = 1):
        '''loads and cleans data.

        Parameters
        ----------
        file : str, optional (defalut = None)
            the name of the data file containing the data.  Must have columns: `Date`, `UT`, and `OutTemp`.  If file = None, then it will automatically be reset to 'k1_temp_mirror_5min.csv'.
        fraction_data : float, optional (defalut = 1)
            the percentage of the data that you want to clean and parse into X and y values.  This value should be in the range of 0 to 1 (not including 0).
        '''
        if file == None:
            self.file = 'k1_temp_mirror_5min.csv'
        else:
            self.file = file

        try:
            dataSet = pd.read_csv(self.file)
        except:
            try:
                zf = zipfile.ZipFile(self.file+'.zip')
                df = pd.read_csv(zf.open(self.file))
            except:
                print('The file \'%s\' does not exist!'%(self.file))

        if not isinstance(fraction_data, float):
            print('fraction_data needs to be a float between 0 and 1 (not including 0)')
            print('setting to default of fraction_data = 1')
            fraction_data = 1

        dataSet.columns = dataSet.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        for i in range(len(dataSet['OutTemp'])):
            vals = [i-2,i-1,i+1,i+2]
            x = np.array(vals)-i
            if -17 > dataSet['OutTemp'][i] or dataSet['OutTemp'][i] > 17:
                y = dataSet['OutTemp'][vals]
                if -17 < np.min(y) and np.max(y) < 17:
                    tck = interpolate.splrep(x, y, s=0)
                    new_x = interpolate.splev(0, tck, der=0)
                    dataSet.at[i, 'OutTemp'] = new_x

        dataSet = dataSet.drop(dataSet[dataSet['OutTemp'] < -17].index).reset_index(drop=True)
        dataSet = dataSet.drop(dataSet[dataSet['OutTemp'] > 17].index).reset_index(drop=True)

        format = '%Y-%m-%d %H:%M'
        dataSet['Datetime'] = pd.to_datetime(dataSet['Date'] + dataSet['UT'], format=format)
        dataSet['year'] = pd.DatetimeIndex(dataSet['Datetime']).year
        dataSet['month'] = pd.DatetimeIndex(dataSet['Datetime']).month
        dataSet['day'] = pd.DatetimeIndex(dataSet['Datetime']).day
        dataSet['hour'] = pd.DatetimeIndex(dataSet['Datetime']).hour
        dataSet['minute'] = pd.DatetimeIndex(dataSet['Datetime']).minute


        cutoff = int(len(dataSet)*fraction_data)
        if fraction_data < 1:
            print('Number of rows: %s'%(cutoff))
        if fraction_data > 1:
            fraction_data = 1
            print('fraction_data above 1. Setting fraction_data = 1')

        if len(dataSet) > 10000 and cutoff > 1000:
            self.df = dataSet[:cutoff]
        elif cutoff < 1000:
            self.df = dataSet[:1000]
            print('fraction_data too small, taking 1000 rows instead')
        else:
            self.df = dataSet
            print('Data set too small')

    def create(self, hours_ahead, test_size = 0.1):
        '''cleans and reshapes data into usable machine learning matricies X.

        Parameters
        ----------
        hours_ahead : int
            the value of how many hours ahead we are trying to predict.  The value must be an int between 4 and 8 (including both 4 and 8).
        test_size : float, optional (default = 0.1)
            the value of what percentage of X and y should be the saved for the testing sets.  The value should be between 0 and 1 (not including 0).
        '''
        if not isinstance(hours_ahead, int):
            print('hours_ahead needs to be an integer!\n Trying to make int...')
            try:
                hours_ahead = int(hours_ahead)
                print('hours_ahead = %s'%(hours_ahead))
            except:
                print('Failed!')
                import sys
                sys.exit(1)

        if not isinstance(test_size, float):
            print('test_size needs to be a float between 0 and 1 (not including 0)')
            print('setting to default of test_size = 0.1')
            test_size = 0.1

        if hours_ahead > 8:
            print('Can\'t be greater than 8, so resetting hours_ahead to 8')
            hours_ahead = 8
        elif hours_ahead < 4:
            print('Can\'t be less than 4, so resetting hours_ahead to 4')
            hours_ahead = 4



        self.n_temp_predictors = (24-hours_ahead)*10
        self.n_predictors = self.n_temp_predictors+4
        self.hours_ahead = hours_ahead
        values_ahead = hours_ahead*10

        data = self.df['OutTemp']
        time = self.df['UT']
        date = self.df['Date']
        length = len(data)
        X = []
        y = []
        y_date = []
        pbar = ProgressBar()
        for i in pbar(range(0,length-self.n_temp_predictors-values_ahead)):
            x = []
            t = []
            d = []
            labels = []

            if 3 <= self.df.hour[i+self.n_temp_predictors+values_ahead-1] <= 7:
                temp_hour = self.df.hour[i+self.n_temp_predictors+values_ahead-1]

                for j in range(0, self.n_temp_predictors):
                    string = date[i+j]+' '+time[i+j]
                    date1 = datetime.strptime(string, '%Y-%m-%d %H:%M')
                    d.append(date1)

                    x.append(data[i+j])
                    labels.append('%s_hr_ago'%((j+1)*6/60))

                import csaps
                t = np.array(range(len(x)))*6
                sp0 = csaps.UnivariateCubicSmoothingSpline(t, x, smooth=0.01)
                xs0 = np.linspace(t[0], t[-1], self.n_temp_predictors)
                x = sp0(xs0)


                y_temp = data[i+self.n_temp_predictors+values_ahead-1]

                date_x_prob = np.sum(np.diff(d) != timedelta(minutes=6))
                string = date[i+self.n_temp_predictors+values_ahead-1]+' '+time[i+self.n_temp_predictors+values_ahead-1]
                date_y = datetime.strptime(string, '%Y-%m-%d %H:%M')
                date_y_prob = np.sum(np.diff([d[-1], date_y]) != timedelta(minutes=values_ahead*6))

                x = np.append(x, self.df.month[i+self.n_temp_predictors+values_ahead-1])
                x = np.append(x, self.df.day[i+self.n_temp_predictors+values_ahead-1])
                x = np.append(x, self.df.hour[i+self.n_temp_predictors+values_ahead-1])
                x = np.append(x, self.df.minute[i+self.n_temp_predictors+values_ahead-1])
                labels.append('Month')
                labels.append('Day')
                labels.append('Hour')
                labels.append('Minute')
                if date_x_prob + date_y_prob == 0:
                    X.append(x)
                    y.append(y_temp)
                    y_date.append(date_y)

        self.X = np.array(X)
        self.y = np.array(y)
        self.y_date = np.array(y_date)
        print(self.X.shape)

        if test_size*len(y) < 30:
            test_size = 30
        elif test_size*len(y) > 7200:
            test_size = 7200

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)

        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.labels = np.array(labels)
