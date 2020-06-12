import sys
sys.path.insert(0, './functions.py')
## Saving print statements to a log, saving figures in /img/
log = open("scoring.log", "a")
sys.stdout = log

from functions import *

model_df = pd.read_csv('./data/interview_data_prepped.csv')

X = model_df.drop(columns = ['read_rate', 'read_rate_discrete', 'read_rate_discrete_even'])
y = model_df['read_rate_discrete']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# For comparison of even bins versus quartile bins
y_2 = model_df['read_rate_discrete_even']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_2, test_size = 0.3, random_state = 1, stratify = y)

rf_model = joblib.load('./models/Random_Forest_v1.joblib')
rf_model_even_bins = joblib.load('./models/Random_Forest_EvenBins_v1.joblib')
lr_model = joblib.load('./models/LogReg_v1.joblib')
lr_model_wpca = joblib.load('./models/LogReg_w_PCA_v1.joblib')

class_labels = ['High', 'Low', 'Med']

print("Random Forest Scores:\n")
report_model_scores(rf_model, X_train, X_test, y_train, y_test)
plot_confusion_matrix(confusion_matrix(y_test, rf_model.predict(X_test)), title='RF_Cnf_Mtrx', img_name = 'RF_v1', classes = class_labels, normalize=True)

print("Random Forest Score (Even bins)\n")
report_model_scores(rf_model_even_bins, X_train2, X_test2, y_train2, y_test2)
plot_confusion_matrix(confusion_matrix(y_test, rf_model_even_bins.predict(X_test)), title='RF_EB_Cnf_Mtrx', img_name = 'RF_EB_v1', classes = class_labels, normalize=True)

print("Log Reg Scores:\n")
report_model_scores(lr_model, X_train, X_test, y_train, y_test)
plot_confusion_matrix(confusion_matrix(y_test, lr_model.predict(X_test)), title='LogReg_Mtrx', img_name = 'LogReg_v1', classes = class_labels, normalize=True)

print("Log Reg with PCA Scores:\n")
report_model_scores(lr_model_wpca, X_train, X_test, y_train, y_test)
plot_confusion_matrix(confusion_matrix(y_test, lr_model_wpca.predict(X_test)), title='LogRegPCA_Mtrx', img_name = 'LogReg_w_PCA_v1', classes = class_labels, normalize=True)


### Exporting Feature coefficients/importances using best parameters obtained
### from model optimization.
lr_export = LogisticRegression(C =100, penalty = 'l2')
lr_export.fit(X_train, y_train)
lr_coeffs = pd.DataFrame(data = {
        "Feature": X.columns,
        "High":lr_export.coef_[0]/max(lr_export.coef_[0])*1000,
        "Low" :lr_export.coef_[1]/max(lr_export.coef_[1])*1000,
        "Med" :lr_export.coef_[2]/max(lr_export.coef_[2])*1000
    },index = range(0,len(lr_export.coef_[0]))  )
lr_coeffs.to_csv('./temp/logreg_normalized_features.csv', index = False)
print("\nLogReg Coeffs Table: ", lr_coeffs)

dc_export = DecisionTreeClassifier(max_features=0.5, max_depth=11, criterion='entropy',min_samples_split=4)
dc_export.fit(X_train, y_train)
dc_fi_table = pd.DataFrame(data = {
        "Feature": X.columns,
        "Importance":dc_export.feature_importances_/max(dc_export.feature_importances_)*100,
    },index = range(0,len(lr_export.coef_[0]))  )
dc_fi_table.to_csv('./temp/dc_feature_importances.csv', index = False)
print("\nDC/RF Feature importances: ", dc_fi_table)
