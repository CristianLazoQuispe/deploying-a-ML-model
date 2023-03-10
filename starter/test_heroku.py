"""
Heroku test
"""
import requests
import json

data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

response = requests.post('https://homework-deploying-ml-fastapi.herokuapp.com/predict/', data=json.dumps(data))

print("Response code: %s" % response.status_code)
print("Response body: %s" % response.json())

assert response.status_code == 200
assert json.loads(response.text)["forecast"] == "Income < 50k"

