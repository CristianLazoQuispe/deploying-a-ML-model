# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- We used a Random Forest Classifier
- Version 1.0.0
- Created on 25/01/2023
- For more information about the model, please refer to Udacity Nanodegree program


## Intended Use
- This project is intended to solve a project of mlops deploying a Scalable ML Pipeline in Production
- The target is to predict if an income exceeds $50K/yr based on census data


## Factors
- The data is from UCI (https://archive-beta.ics.uci.edu/datasets?search=Census+Income++)
- This data about census data

## Metrics
We used metrics: precision_score, recall_score,  and fbeta_score. Performance results for the slices of education feature is presented in this table:


    Bachelors       Precision : 0.9599  Recall : 0.9496  Beta : 0.9547
    HS-grad         Precision : 0.9357  Recall : 0.8693  Beta : 0.9013
    11th            Precision : 1.0000  Recall : 0.7833  Beta : 0.8785
    Masters         Precision : 0.9598  Recall : 0.9708  Beta : 0.9653
    9th             Precision : 1.0000  Recall : 0.8519  Beta : 0.9200
    Some-college    Precision : 0.9484  Recall : 0.9012  Beta : 0.9242
    Assoc-acdm      Precision : 0.9522  Recall : 0.9019  Beta : 0.9264
    Assoc-voc       Precision : 0.9534  Recall : 0.9058  Beta : 0.9290
    7th-8th         Precision : 1.0000  Recall : 0.8000  Beta : 0.8889
    Doctorate       Precision : 0.9646  Recall : 0.9804  Beta : 0.9724
    Prof-school     Precision : 0.9650  Recall : 0.9787  Beta : 0.9718
    5th-6th         Precision : 1.0000  Recall : 0.6250  Beta : 0.7692
    10th            Precision : 0.9508  Recall : 0.9355  Beta : 0.9431
    1st-4th         Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
    Preschool       Precision : 0.0000  Recall : 1.0000  Beta : 0.0000
    12th            Precision : 1.0000  Recall : 0.8182  Beta : 0.9000

## Evaluation Data
We used 20% of the dataset for evaluation purposes of the model.

## Training Data
We used 80% of the dataset for the training purposes of the model.

## Quantitative Analyses
Model performance is measured with the overall metrics and slice performance analysis

## Ethical Considerations

According to the data slices, the model have a underperformance in Preschool. This is a indicator os discriminate people because the prediction is always "Income < 50k". This observation requires futher investigation.

<img src = "starter/results/slicer_performance.png?raw=true" width = "700" height = "300" />

## Caveats and Recommendations
The data is imbalance and biased and need to be investigated more. For example in the Preschool category as we can see in the previos step.