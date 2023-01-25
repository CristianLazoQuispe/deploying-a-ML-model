# Put the code for your API here.
import os
import pandas as pd

# Import libraries related to fastAPI
from fastapi import FastAPI
from pydantic import BaseModel

# Import the inference function to be used to predict the values
try:
    from starter.starter.ml.model import inference
    from starter.starter.ml.data import process_data
except ImportError:
    from starter.ml.model import inference
    from starter.ml.data import process_data


# Import the model to be used to predict
model = pd.read_pickle("starter/model/model.pkl")
encoder = pd.read_pickle("starter/model/encoder.pkl")

# Initial a FastAPI instance
app = FastAPI()

# Give Heroku the ability to pull in data from DVC upon app start up.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# pydantic models


class DataIn(BaseModel):
    # The input should be alist of 108 values
    age: int = 39
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = 2174
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "United-States"


class DataOut(BaseModel):
    # The forecast output will be either >50K or <50K
    forecast: str = "Income > 50k"

# Adding GET  Welcome message to the initial page


@app.get("/welcome")
async def root():
    return {"Welcome": "Welcome to the ML project!"}


# Adding POST predict
@app.post("/predict", response_model=DataOut, status_code=200)
def get_prediction(payload: DataIn):
    # Converted into dataframe
    df = pd.DataFrame([{"age": payload.age,
                        "workclass": payload.workclass,
                        "fnlgt": payload.fnlgt,
                        "education": payload.education,
                        "education-num": payload.education_num,
                        "marital-status": payload.marital_status,
                        "occupation": payload.occupation,
                        "relationship": payload.relationship,
                        "race": payload.race,
                        "sex": payload.sex,
                        "capital-gain": payload.capital_gain,
                        "capital-loss": payload.capital_loss,
                        "hours-per-week": payload.hours_per_week,
                        "native-country": payload.native_country}])

    # Process the data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    X_processed, _, _, _ = process_data(
        df, categorical_features=cat_features, training=False, encoder=encoder)

    # Make an inference
    prediction = inference(model, X_processed)

    # Converted into income format
    if prediction == 1:
        prediction = "Income > 50k"
    elif prediction == 0:
        prediction = "Income < 50k"

    # Return prediction
    response_object = {"forecast": prediction}
    return response_object
