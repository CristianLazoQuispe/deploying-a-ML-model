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


        Training metrics: precision=0.9990 recall=0.9957 fbeta=0.9974
        Testing  metrics: precision=0.7420 recall=0.6225 fbeta=0.6771


        Training dataset    Precision : 0.9990  Recall : 0.9957  Beta : 0.9974
        --------------------------------------------------------
        Bachelors       Precision : 0.9983  Recall : 0.9972  Beta : 0.9977
        HS-grad         Precision : 0.9985  Recall : 0.9925  Beta : 0.9955
        11th            Precision : 1.0000  Recall : 0.9796  Beta : 0.9897
        Masters         Precision : 0.9987  Recall : 0.9987  Beta : 0.9987
        9th             Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        Some-college    Precision : 1.0000  Recall : 0.9928  Beta : 0.9964
        Assoc-acdm      Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        Assoc-voc       Precision : 1.0000  Recall : 0.9966  Beta : 0.9983
        7th-8th         Precision : 1.0000  Recall : 0.9706  Beta : 0.9851
        Doctorate       Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        Prof-school     Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        5th-6th         Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        10th            Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        1st-4th         Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        Preschool       Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        12th            Precision : 1.0000  Recall : 1.0000  Beta : 1.0000


        Testing dataset    Precision : 0.7420  Recall : 0.6225  Beta : 0.6771
        --------------------------------------------------------
        Bachelors       Precision : 0.7714  Recall : 0.7200  Beta : 0.7448
        HS-grad         Precision : 0.6407  Recall : 0.4290  Beta : 0.5139
        11th            Precision : 0.8000  Recall : 0.3636  Beta : 0.5000
        Masters         Precision : 0.8279  Recall : 0.8599  Beta : 0.8436
        9th             Precision : 1.0000  Recall : 0.3333  Beta : 0.5000
        Some-college    Precision : 0.6584  Recall : 0.4801  Beta : 0.5553
        Assoc-acdm      Precision : 0.7222  Recall : 0.5532  Beta : 0.6265
        Assoc-voc       Precision : 0.6739  Recall : 0.4921  Beta : 0.5688
        7th-8th         Precision : 0.0000  Recall : 0.0000  Beta : 0.0000
        Doctorate       Precision : 0.8596  Recall : 0.8596  Beta : 0.8596
        Prof-school     Precision : 0.8298  Recall : 0.9286  Beta : 0.8764
        5th-6th         Precision : 1.0000  Recall : 0.5000  Beta : 0.6667
        10th            Precision : 0.3333  Recall : 0.1667  Beta : 0.2222
        1st-4th         Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        Preschool       Precision : 1.0000  Recall : 1.0000  Beta : 1.0000
        12th            Precision : 1.0000  Recall : 0.4000  Beta : 0.5714
        12th            Precision : 1.0000  Recall : 0.8182  Beta : 0.9000


## Evaluation Data
We used 20% of the dataset for evaluation purposes of the model.

## Training Data
We used 80% of the dataset for the training purposes of the model.

## Quantitative Analyses
Model performance is measured with the overall metrics and slice performance analysis

## Ethical Considerations

According to the data slices, the model have a underperformance in 7th-8th. This is a indicator os discriminate people because the prediction is always "Income < 50k". This observation requires futher investigation.

Performance in train

<img src = "results/slicer_performance_education_train.png?raw=true" width = "700" height = "300" />

Performance in test

<img src = "results/slicer_performance_education_test.png?raw=true" width = "700" height = "300" />


## Caveats and Recommendations
The data is imbalance and biased and need to be investigated more. For example in the Preschool category as we can see in the previos step.