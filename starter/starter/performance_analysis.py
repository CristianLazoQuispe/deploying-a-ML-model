import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


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
    data = pd.read_csv("starter/data/census_cleaned.csv")
    model = pd.read_pickle("starter/model/model.pkl")
    encoder = pd.read_pickle("starter/model/encoder.pkl")
    lb = pd.read_pickle("starter/model/lb.pkl")

    return data, model, encoder, lb


def compute_performance_slice(model,encoder,lb,data,slice_name,slice_classes,suffix=''):
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


    # Write the performance in txt
    txt_file = open(
        'starter/results/{}_performance{}.txt'.format(slice_name,suffix),
        'w')
    txt_file.write(
        'Performance results for the slices of {} feature'.format(slice_name))
    txt_file.write("\n")
    
    X_test, y_test, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    txt_file.write("\n")
    txt_file.write(
        '{:15s}  {:10s} {:1.4f}'.format(
            "Total dataset",
            "Precision :",
            precision))
    txt_file.write('  {:7s} {:1.4f}'.format("Recall :", recall))
    txt_file.write('  {:5s} {:1.4f}'.format("Beta :", fbeta))
    txt_file.write("\n")
    txt_file.write('--------------------------------------------------------')


    list_precision = []
    for slice_class in slice_classes:
        df = data.loc[data.loc[:, slice_name] == slice_class, :]
        X_test, y_test, encoder, lb = process_data(
            df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        txt_file.write("\n")
        txt_file.write(
            '{:15s}  {:10s} {:1.4f}'.format(
                slice_class,
                "Precision :",
                precision))
        txt_file.write('  {:7s} {:1.4f}'.format("Recall :", recall))
        txt_file.write('  {:5s} {:1.4f}'.format("Beta :", fbeta))
        list_precision.append(precision)
    txt_file.close()

    # Figure Size
    plt.figure(figsize=(20, 7))
    # Horizontal Bar Plot
    plt.bar(slice_classes, list_precision)
    plt.title("Slicer performance precision "+suffix)
    # Save plot
    plt.savefig("starter/results/slicer_performance"+suffix+".png", bbox_inches='tight')
    plt.close()

def get_performance():
    """
    Computes slicer performance

    Returns
    -------
    None
    """

    # Reading model artifacts
    data, model, encoder, lb = get_artifacts()

    # Split dataset
    train, test = train_test_split(data, test_size=0.20,random_state=42)

    # Unique feature of education
    education_classes = data.loc[:, "education"].unique()

    #compute performance in train and test
    compute_performance_slice(model,encoder,lb,train,"education",education_classes,suffix='_education_train')
    compute_performance_slice(model,encoder,lb,test,"education",education_classes,suffix='_education_test')

    # Unique feature of sex
    sex_classes = data.loc[:, "sex"].unique()

    #compute performance in train and test
    compute_performance_slice(model,encoder,lb,train,"sex",sex_classes,suffix='_sex_train')
    compute_performance_slice(model,encoder,lb,test,"sex",sex_classes,suffix='_sex_test')


if __name__ == "__main__":
    get_performance()
