# Deploying a ML model using FastAPI and Heroku

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.


# Links


Github: https://github.com/CristianLazoQuispe/deploying-a-ML-model

Heroku: https://homework-deploying-ml-fastapi.herokuapp.com/welcome


# Environment Set up
* Download and install conda if you don’t have it already.
* Use the supplied requirements file to create a new environment, or
* conda create - n[envname] "python=3.8" scikit - learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn - c conda - forge
* Install git either through conda(“conda install git”) or through your CLI, e.g. sudo apt - get git.

# Repositories
* Create a directory for the project and initialize git.
* As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre - made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
* Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

I have explorated the data in a "notebooks/Exploration Data Analysis.ipynb"


# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.

    $ python starter/starter/train_model.py

            Training metrics: precision=0.9990 recall=0.9957 fbeta=0.9974
            Testing  metrics: precision=0.7420 recall=0.6225 fbeta=0.6771

* Write unit tests for at least 3 functions in the model code.

    $ pytest starter/test_ml.py -v


<img src = "starter/screenshots/unit_test.png?raw=true" width = "900" height = "200" />


* Write a function that outputs the performance of the model on slices of the data.
* Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.

    $ python starter/starter/performance_analysis.py 

Performance in train

<img src = "starter/results/slicer_performance_education_train.png?raw=true" width = "700" height = "300" />

Performance in test

<img src = "starter/results/slicer_performance_education_test.png?raw=true" width = "700" height = "300" />

* Write a model card using the provided template.


# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI / Pydantic / etc to deal with this.
* Write 3 unit tests to test the API(one for the GET and two for POST, one that tests each prediction).

    $ uvicorn starter.main:app --reload
    $ pytest starter/test_api.py -v

<img src = "starter/screenshots/API_unit_test.png?raw=true" width = "1000" height = "300" />

* Run sanity check for your test cases:

    * Run python sanitycheck.py. This script is located inside the starter directory in the starter code.
    * The script will scan the test cases written for the GET() and POST() APIs and generate a report.
    * The report will list any problems it detects with your test cases. Fix the problems and run the sanitycheck.py script again.
    * The script uses heuristics to detect common problems and can sometimes overlook a problem or raise a false alarm. You should still check your implementation against the project rubric to be absolutely sure your submission will meet the requirements.

    $ python starter/sanitycheck.py

        > starter/test_api.py


<img src = "starter/screenshots/API_sanity_check.png?raw=true" width = "1000" height = "300" />

# API Deployment
* Create a free Heroku account(for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
* Enable automatic deployments that only deploy if your continuous integration passes.
* Hint: think about how paths will differ in your local environment vs. on Heroku.
* Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI / CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.


<img src = "starter/screenshots/deployment_heroku.png?raw=true" width ="700" height = "500" />


    $ python starter/test_heroku.py 


<img src = "starter/screenshots/API_heroku_response.png?raw=true" width = "400" height = "300" />