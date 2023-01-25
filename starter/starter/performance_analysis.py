import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics
import numpy as np
from matplotlib import pyplot as plt
 
def get_artifacts():
    """
    Read artifacts from source
    
    Returns
    ------
    
    data : pd.DataFrame
        census dataframe
    model: sklearn.ensemble._forest.RandomForestClassifier
        random forest classifier
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer
    """    
    # Add code to load in the data, model and encoder
    data    = pd.read_csv("starter/data/census_cleaned.csv")
    model   = pd.read_pickle("starter/model/model.pkl")
    encoder = pd.read_pickle("starter/model/encoder.pkl") 
    lb      = pd.read_pickle("starter/model/lb.pkl")

    return data, model, encoder, lb

def get_performance():
    """ 
    Computes slicer performance
    
    Returns
    -------
    None
    """
    
    # Reading model artifacts
    data, model, encoder, lb = get_artifacts()

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    #Unique feature of education
    education_classes = data.loc[:,"education"].unique()

    #Write the performance in txt
    txt_file = open('starter/results/{}_performance.txt'.format("education"), 'w')
    txt_file.write('Performance results for the slices of {} feature'.format("education"))
    txt_file.write("\n")

    list_precision = []
    for education in education_classes:
        df = data.loc[data.loc[:,"education"]==education,:]
        X_test, y_test, encoder, lb = process_data(
        df, categorical_features=cat_features, label="salary", training=False, encoder=encoder,lb=lb)
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        txt_file.write("\n")
        txt_file.write('{:15s}  {:10s} {:1.4f}'.format(education,"Precision :",precision))
        txt_file.write('  {:7s} {:1.4f}'.format("Recall :",recall))
        txt_file.write('  {:5s} {:1.4f}'.format("Beta :",fbeta))
        list_precision.append(precision)
    txt_file.close()


    # Figure Size
    fig = plt.figure(figsize =(20, 7))    
    # Horizontal Bar Plot
    plt.bar(education_classes, list_precision)    
    plt.title("Slicer performance precision")
    # Save plot
    plt.savefig("starter/results/slicer_performance.png", bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    get_performance()