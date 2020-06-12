import sys
sys.path.insert(0, './functions.py')

from functions import *

### Loading in data, performing TTS ###
model_df = pd.read_csv('./data/interview_data_prepped.csv')

X = model_df.drop(columns = ['read_rate', 'read_rate_discrete', 'read_rate_discrete_even'])
y = model_df['read_rate_discrete']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

### Model Training ###
rf_params = {
    "n_estimators" : [50, 100, 500],
    "criterion"    : ['entropy', 'gini'],
    "max_depth"    : [3, 7, 11],
    "min_samples_split" : [2, 4, 6],
    "max_features" : [0.5, 0.75, 1.0]
}

rf_model = train_random_forest(X_train, y_train, input_model_parameters = rf_params, model_name_save = 'Random_Forest_v1')


logreg_params = {

    "lr__penalty"       : ["l1", 'l2'],
    "lr__C"             : np.logspace(-3,3,7)

}

lr_model = train_log_reg(X_train, y_train, logreg_params, model_name_save = 'LogReg_v1', use_pca = False)


logreg_params_wpca = {

    "pca__n_components" : [4,7,10,15],
    "lr__penalty"       : ["l1", 'l2'],
    "lr__C"             : np.logspace(-3,3,7)

}

lr_model_wpca = train_log_reg(X_train, y_train, logreg_params_wpca, model_name_save = 'LogReg_w_PCA_v1', use_pca = True)

# For comparison of even bins versus quartile bins
y_2 = model_df['read_rate_discrete_even']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_2, test_size = 0.3, random_state = 1, stratify = y)
rf_model_even_bins = train_random_forest(X_train2, y_train2, input_model_parameters = rf_params, model_name_save = 'Random_Forest_EvenBins_v1')

# Defining a log file to save outputs, comment out to print in terminal normally
log = open("training.log", "a")
sys.stdout = log


print("Random Forest Scores:\n")
report_model_scores(rf_model, X_train, X_test, y_train, y_test)

print("Random Forest Score (Even bins)\n")
report_model_scores(rf_model_even_bins, X_train2, X_test2, y_train2, y_test2)

print("Log Reg Scores:\n")
report_model_scores(lr_model, X_train, X_test, y_train, y_test)

print("Log Reg with PCA Scores:\n")
report_model_scores(lr_model_wpca, X_train, X_test, y_train, y_test)
