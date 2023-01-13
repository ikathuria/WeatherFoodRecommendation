import os
import json
import pickle
import requests
import numpy as np
import pandas as pd
from datetime import datetime

from skfuzzy import control as ctrl
from skfuzzy import gaussmf, interp_membership


TIME = ctrl.Antecedent(np.arange(0, 24, 1), 'TIME')
TIME['late_night'] = gaussmf(TIME.universe, 3, 3)
TIME['morning'] = gaussmf(TIME.universe, 9.5, 2)
TIME['afternoon'] = gaussmf(TIME.universe, 14.5, 1.5)
TIME['evening'] = gaussmf(TIME.universe, 17.5, 1.5)
TIME['night'] = gaussmf(TIME.universe, 21.5, 2)


def weather_api(lat, long):
    """
    Get realtime weather from openweathermap.org.
    """

    key = "7f9385228d95a55a108efe83bee7fa24"
    api = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={long}&appid={key}&units=metric"
    response = requests.get(api)

    return json.loads(response.text)


def get_realtime_weather(now, lat, long):
    """
    Check if realtime weather needs to be updated.
    """

    file_mod = datetime.fromtimestamp(os.path.getmtime(
        "data/realtime_weather.json"
    ))

    # check if current file is older than 30 mins then update
    if abs(file_mod - now).seconds / 60 > 30:
        rt_weather = weather_api(lat, long)

        with open("realtime_weather.json", "w") as f:
            json.dump(rt_weather, f)

    else:
        with open("realtime_weather.json", "r") as f:
            rt_weather = json.load(f)

    return rt_weather


def get_weather_fields(weather_data):
    """
    Current weather fields.
    """
    temp_avg = temp_min = temp_max = prec = 0.0
    try:
        temp_avg = weather_data['main']['temp']
        temp_min = weather_data['main']['temp_min']
        temp_max = weather_data['main']['temp_max']
    except:
        pass

    try:
        prec = weather_data['rain']['3h']
    except:
        try:
            prec = weather_data['rain']['1h']
        except:
            pass

    return temp_avg, temp_min, temp_max, prec


def fuzzify_input(curr_time):
    """
    Fuzzify current time to get meal time.
    """
    late_night_fuzzy = interp_membership(
        TIME.universe, TIME['late_night'].mf, curr_time
    )
    morning_fuzzy = interp_membership(
        TIME.universe, TIME['morning'].mf, curr_time
    )
    afternoon_fuzzy = interp_membership(
        TIME.universe, TIME['afternoon'].mf, curr_time
    )
    evening_fuzzy = interp_membership(
        TIME.universe, TIME['evening'].mf, curr_time
    )
    night_fuzzy = interp_membership(
        TIME.universe, TIME['night'].mf, curr_time
    )

    meal = {
        'late_night': late_night_fuzzy, 'morning': morning_fuzzy,
        'afternoon': afternoon_fuzzy, 'evening': evening_fuzzy,
        'night': night_fuzzy
    }

    meal = max(meal, key=lambda x: meal[x])

    return meal


def load_model(path):
    """
    Load the pickled Machine Learning model.
    """
    return pickle.load(open(path, 'rb'))


def get_preds(model_path, model_input):
    """
    Get predictions from the model.
    """
    model = load_model(model_path)
    preds = model.predict([model_input])

    with open("data/food_keys.pickle", "rb") as handle:
        food_keys = pickle.load(handle)

    predicitions = {}
    for key, food in food_keys.items():
        predicitions[food] = preds[0][key]

    predicitions = dict(sorted(
        predicitions.items(),
        key=lambda item: item[1], reverse=True
    ))

    return {k: predicitions[k] for k in list(predicitions) if predicitions[k] > 0}



def get_results(latitude=28.7041, longitude=77.1025, model_path="model/gb.sav"):
    """
    Get final food recommendations.
    """
    now = datetime.now()
    now_str = now.strftime("%Y,%m,%d,%H,%M,%S").split(',')
    year, month, date, hour, mins, secs = [int(i) for i in now_str]

    # get current weather
    RT_WEATHER_DATA = get_realtime_weather(now, latitude, longitude)
    TEMP_AVG, TEMP_MIN, TEMP_MAX, PREC = get_weather_fields(RT_WEATHER_DATA)

    # get predictions
    predictions = get_preds(model_path, [TEMP_AVG, TEMP_MIN, TEMP_MAX, PREC])

    # get meal time
    MEAL = fuzzify_input(hour)

    # filter predictions based on meal time
    food_meal_times = pd.read_excel(
        "data/food_meal.xlsx", index_col=0
    ).to_dict('index')

    result = {}
    for food, pred in predictions.items():
        if food_meal_times[food][MEAL] == 1:
            result[food] = pred

    return {k: result[k] for k in list(result)[:10] if result[k] > 0}, TEMP_AVG, MEAL


def get_custom_weather_preds(custom_inputs, latitude=28.7041, longitude=77.1025, model_path="model/gb.sav"):
    """
    Custom weather prediction.
    """

    temp_avg = custom_inputs['temp_avg']
    temp_min = custom_inputs['temp_min']
    temp_max = custom_inputs['temp_max']
    prec = custom_inputs['prec']
    meal = custom_inputs['meal']

    now = datetime.now()
    now_str = now.strftime("%Y,%m,%d,%H,%M,%S").split(',')
    year, month, date, hour, mins, secs = [int(i) for i in now_str]

    if temp_avg == None or temp_min == None or temp_max == None or prec == None:
        weather_data = get_realtime_weather(now, latitude, longitude)
        calc_temp_avg, calc_temp_min, calc_temp_max, calc_prec = get_weather_fields(
            weather_data
        )
        
        if temp_avg == None:
            temp_avg = calc_temp_avg

        if temp_min == None:
            temp_min = calc_temp_min

        if temp_max == None:
            temp_max = calc_temp_max

        if prec == None:
            prec = calc_prec

    else:
        temp_avg = float(temp_avg)
        temp_min = float(temp_min)
        temp_max = float(temp_max)
        prec = float(prec)

    if meal == None:
        meal = fuzzify_input(hour)

    # get predictions
    predictions = get_preds(model_path, [temp_avg, temp_min, temp_max, prec])

    # filter predictions based on meal time
    food_meal_times = pd.read_excel(
        "data/food_meal.xlsx", index_col=0
    ).to_dict('index')

    result = {}
    for food, pred in predictions.items():
        if food_meal_times[food][meal] == 1:
            result[food] = pred

    return {k: result[k] for k in list(result)[:10] if result[k] > 0}, temp_avg, meal
