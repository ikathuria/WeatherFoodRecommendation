import os
import json
from datetime import datetime

from geopy.geocoders import Nominatim

from flask import Flask
from flask import request, render_template, url_for, Response, make_response
from flask import redirect, send_from_directory

from utils import *


APP = Flask(__name__)

# ML model path
model_path = "model/gb.sav"


@APP.route("/", methods=["GET", "POST"])
def home():
    top_10 = temp_avg = meal = None

    if request.method == 'POST':
        top_10, temp_avg, meal = get_results(model_path=model_path)

        return render_template(
            'index.html', prediction=top_10,
            temp_avg=temp_avg, meal=meal
        )

    return render_template(
        'index.html', prediction=None,
        temp_avg=None, meal=None
    )


@APP.route("/custom-weather", methods=["GET", "POST"])
def custom_weather():
    top_10 = temp_avg = meal = None

    if request.method == 'POST':
        custom_inputs = {
            "temp_avg": request.form.get("temp_avg") or None,
            "temp_min": request.form.get("temp_min") or None,
            "temp_max": request.form.get("temp_max") or None,
            "prec": request.form.get("prec") or None,
            "meal": request.form.get("meal") or None
        }

        top_10, temp_avg, meal = get_custom_weather_preds(
            custom_inputs=custom_inputs,
            model_path=model_path
        )

        return render_template(
            'custom_weather.html', prediction=top_10,
            temp_avg=temp_avg, meal=meal
        )

    return render_template(
        'custom_weather.html', prediction=None,
        temp_avg=None, meal=None
    )


if __name__ == "__main__":
    APP.config["ENV"] = "development"
    APP.run(debug=True)
