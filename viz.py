import os
import json
import requests
from datetime import datetime
from utils import *


if __name__ == "__main__":
    RT_WEATHER = "data/realtime_weather.json"

    # Delhi latitude and longitude
    LAT = 28.7041
    LONG = 77.1025

    MODEL = load_model("model/gb.sav")

    now = datetime.now()
    # time_now = now.strftime("%H:%M:%S")
    now_str = now.strftime("%Y,%m,%d,%H,%M,%S").split(',')
    year, month, date, hour, mins, secs = [int(i) for i in now_str]

    file_mod = datetime.fromtimestamp(os.path.getmtime(
        RT_WEATHER
    ))

    if abs(file_mod - now).seconds / 60 > 0:
        RT_WEATHER_DATA = get_realtime_weather(LAT, LONG)
        with open("realtime_weather.json", "w") as f:
            json.dump(RT_WEATHER_DATA, f)
    else:
        with open("realtime_weather.json", "r") as f:
            RT_WEATHER_DATA = json.load(f)

    # current weather
    temp_avg = temp_min = temp_max = prec = 0.0
    try:
        temp_avg = RT_WEATHER_DATA['main']['temp']
        temp_min = RT_WEATHER_DATA['main']['temp_min']
        temp_max = RT_WEATHER_DATA['main']['temp_max']
    except:
        pass

    # current precipitation/rain
    try:
        prec = RT_WEATHER_DATA['rain']['3h']
    except:
        try:
            prec = RT_WEATHER_DATA['rain']['1h']
        except:
            pass

    # get meal time
    meal = fuzzify_input(hour)

    predictions = get_results(MODEL, temp_avg, temp_min, temp_max, prec, meal)

    print("\nTOP 10 RECOMMENDATIONS\n")
    for food, pred in predictions.items():
        print(food, pred)
