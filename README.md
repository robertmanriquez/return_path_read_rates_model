## Readme for submission (Robert Manriquez)

This folder contains all of the assets for my submission for the interview code
challenge for a Data Scientist position at Return Path!  This was developed on
Mac OS using Python 3.6.

-- Contents --

**functions.py**
   * This contains all of my custom functions for this analysis, including functions to perform the following tasks:
       - Data Prep and EDA ("get dummies" functions, make_histogram, make_corr_graph, etc)
       - Model Training (train_random_forest, train_log_reg)
       - Model Scoring (report_model_scores, plot_confusion_matrix, etc)
   * The images are saved in /img/.


**Data_Prep_EDA.py**
   * This script performs the initial data prep using interview_data.csv for modeling, then saves it as interview_data_prepped.csv.
   * Both are located in /data/.


**model_training.py**
   * This script loads interview_data_prepped.csv, splits the data using a set random state, then trains two random forest models and two logistic regression models.
   * The models are saved as .joblib files under /models/
   * The print outs are stored in 'training.log'.


**model_scoring.py**
   * Finally, this script is used for scoring the models by loading in the persisted files, then running relevent classification metrics (Accuracy, Precision, Recall, confusion matrix).
   * Results are stored in the log file 'scoring.log'.


**Summary Report.docx**
   * A quick formal write-up of this work and my findings + analysis.
    
    

This workflow can be executed using the terminal command:

```
python Data_Prep_EDA.py && python model_training.py && python model_scoring.py
```

Thank you for reading!

- Rob Manriquez

