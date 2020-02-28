
from temperatureModels import TemperatureModels
import datetime

def test_init():
    print('\nRunning test_init')
    flag = False
    try:
        models = TemperatureModels()
        flag = True
    except:
        flag = False
    if flag and isinstance(models.models, dict):
        for dict_key in models.models.keys():
            if not isinstance(models.models[dict_key], dict):
                flag = False
    print('\nFinished test_init')
    assert flag

def test_predict_date_out():
    print('\nRunning test_predict_date_out')
    models = TemperatureModels()
    date_time_str = '2018-06-29 08:11:11.11111'
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    x = [0]*200
    should_be_none = models.predict(x, date_time_obj)['pred_val']

    print('\nFinished test_predict_date_out')
    assert should_be_none is None

def test_predict_date_in():
    print('\nRunning test_predict_date_in')
    models = TemperatureModels()
    date_time_str = '2018-06-29 23:11:11.11111'
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    x = [0]*200
    should_be_float = models.predict(x, date_time_obj)['pred_val']
    print(should_be_float, type(should_be_float))
    print('\nFinished test_predict_date_in')
    assert isinstance(should_be_float, float)

def test_predict_x_wrong():
    print('\nRunning test_predict_x_wrong')
    models = TemperatureModels()
    date_time_str = '2018-06-29 23:11:11.11111'
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    x = [0]*201
    should_be_none = models.predict(x, date_time_obj)['pred_val']

    print('\nFinished test_predict_x_wrong')
    assert should_be_none is None

def test_predict_all_date_out():
    print('\nRunning test_predict_all_date_out')
    models = TemperatureModels()
    date_time_str = '2018-06-29 08:11:11.11111'
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    x = [0]*200
    should_be_empty_dict = models.predict_all(x, date_time_obj)

    print('\nFinished test_predict_all_date_out')
    assert should_be_empty_dict == {}

def test_predict_all_date_in():
    print('\nRunning test_predict_all_date_in')
    models = TemperatureModels()
    date_time_str = '2018-06-29 23:11:11.11111'
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
    x = [0]*200
    should_be_dict = models.predict_all(x, date_time_obj)
    if isinstance(should_be_dict, dict) and len(list(should_be_dict.keys())) == 5:
        flag = True
    else:
        flag = False

    print('\nFinished test_predict_all_date_in')
    assert flag

def test_update():
    print('\nRunning test_update_all')
    models = TemperatureModels()
    filename = 'k1_temp_mirror_5min.csv'
    should_be_dict = models.update(8, filename, fraction_data = 0.01)
    flag = False
    if isinstance(should_be_dict, dict) and len(should_be_dict['coef']) == 164:
            flag = True
    print('\nFinished test_update_all')
    assert flag

def test_update_all():
    print('\nRunning test_update_all')
    models = TemperatureModels()
    filename = 'k1_temp_mirror_5min.csv'
    should_be_list = models.update_all(filename, fraction_data = 0.01)
    print('\nFinished test_update_all')
    assert len(should_be_list) == 5

def test_save():
    print('\nRunning test_save')
    models_1 = TemperatureModels()
    models_1.save()
    models_2 = TemperatureModels()
    print('\nFinished test_save')
    assert models_1.models == models_2.models
