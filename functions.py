## Functions used for submission.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.externals import joblib

import itertools

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

from subprocess import call
import pydotplus


def read_rate_qcut(x):
    ### This function converts the read_rate target into quartile ranges, the bins
    ### were determined using pd.qcut, but cutting out the zeros first, then
    ### pasting them back in under the "Low" label.
    if x == 0.0:
        return 'Low'
    elif x > 0 and x <= 0.0652:
        return 'Low'
    elif x > 0.0652 and x <= 0.172:
        return 'Med'
    elif x > 0.172:
        return 'High'

def get_top_domain_dummies(input_df):
    ### Converting the nominal category of "Domain_extension" into dummies, then grabbing only the
    ### top 10 most correlated columns with the read_rate target.  The rest are labels as "other"

    domain_dummies_series = pd.get_dummies(data = input_df[['read_rate','Domain_extension']], columns = ['Domain_extension']).corr()['read_rate']
    top_10_domains = domain_dummies_series.abs().sort_values(ascending = False)[1:10].keys().tolist()
    top_10_domains = [i.replace('Domain_extension_','') for i in top_10_domains]

    input_df['Domain_extension'] = input_df['Domain_extension'].map(lambda x: 'DE_' + x if x in top_10_domains else "DE_other")
    domain_dummies = pd.get_dummies(data = input_df.Domain_extension, columns = ['Domain_extension'])

    domain_dummies.to_csv('./temp/domain_dummies.csv', index = False)

    return domain_dummies

def get_top_from_domain_dummies(input_df):
    ### Converting the nominal catergory "from_domain_hash" into dummies, taking only the Top 100 most
    ### populated labels, then taking only the top 10 most correlated with read_rate from that 100.
    ### The rest are labeled as "other"

    top_100_from_domain = input_df.from_domain_hash.value_counts()[:100].keys().tolist()
    input_df['from_domain_hash'] = input_df['from_domain_hash'].map(lambda x: x if x in top_100_from_domain else 'other')

    from_domain_top100_dummies = pd.get_dummies(data = input_df[['read_rate','from_domain_hash']],columns = ['from_domain_hash'])
    top_100_series = from_domain_top100_dummies.corr()['read_rate'].abs().sort_values(ascending = False)

    top_10_from_domain = top_100_series[1:10].keys().tolist()
    top_10_from_domain = [i.replace('from_domain_hash_','') for i in top_10_from_domain]

    input_df.from_domain_hash = input_df.from_domain_hash.map(lambda x: 'FDH_' + x if x in top_10_from_domain else 'FDH_other')

    from_domain_hash_dummies = pd.get_dummies(data = input_df.from_domain_hash, columns = ['from_domain_hash'])

    from_domain_hash_dummies.to_csv('./temp/from_domain_hash_dummies_top10.csv', index = False)

    return from_domain_hash_dummies

def get_dow_dummies(input_df):
    ### Converting Days of the Week into dummy categories
    dow_dummies = pd.get_dummies(data = input_df['day'], columns = ['day'])

    return dow_dummies

def make_corr_graph(input_df, graph_name):

    fig = plt.figure()
    ax = input_df.corr()['read_rate'][2:].sort_values().plot(kind = 'barh', figsize = (7,6), fontsize = 10)
    plt.xlabel('Correlations (0 - 1)', fontsize = 12)
    plt.ylabel('Variable Names', fontsize = 12)
    fig.savefig('../Return_Path_test/img/'+ str(graph_name) , bbox_inches='tight')

def make_histogram(input_series, graph_name):

    fig = plt.figure()
    ax = input_series.hist(bins = 50, figsize = (5,4))
    plt.xlabel('Read Rate (0 - 1)', fontsize = 14)
    plt.ylabel('Counts', fontsize = 12)
    fig.savefig('../Return_Path_test/img/'+ str(graph_name) , bbox_inches='tight')

def train_random_forest(input_X_train, input_y_train, input_model_parameters, model_name_save = 'new_rf'):
    ### Training a Random Forest model, using RandomizedSearchCV to save computation time rather than GridSearch.
    ### Saves the trained model under /models folder.
    rf = RandomForestClassifier(class_weight='balanced')

    rs_rf = RandomizedSearchCV(
        estimator = rf,
        param_distributions = input_model_parameters,
        cv = 3,
        verbose = 3,
        n_jobs = 4,
        scoring = 'f1_micro'
    )

    rs_rf.fit(input_X_train, input_y_train)

    joblib.dump(rs_rf, './models/' + model_name_save + '.joblib')

    return rs_rf


def train_log_reg(input_X_train, input_y_train, input_model_parameters, model_name_save = 'new_logreg', use_pca = False):
    ### Training a LogisticRegression model, using RandomizedSearchCV to save computation time rather than GridSearch.
    ### Has the option to include PCA in the model pipeline.
    ### Saves the trained model under /models folder.

    log_reg = LogisticRegression(class_weight='balanced')

    if use_pca:
        model_pipe = Pipeline(
            [
                ( 'ss'  , StandardScaler() ),
                ( 'pca' , PCA() ),
                ( 'lr'  , log_reg )
            ]
        )

    else:
        model_pipe = Pipeline(
            [
                ( 'ss'  , StandardScaler() ),
                ( 'lr'  , log_reg )
            ]
        )

    rs_lr = RandomizedSearchCV(
        estimator = model_pipe,
        param_distributions = input_model_parameters,
        cv = 3,
        verbose = 1,
        n_jobs = 4,
        scoring = 'f1_macro'
    )

    rs_lr.fit(input_X_train, input_y_train)

    joblib.dump(rs_lr, './models/' + model_name_save + '.joblib')

    return rs_lr

    rs_rf = RandomizedSearchCV(
        estimator = rf,
        param_distributions = input_model_parameters,
        cv = 3,
        verbose = 1,
        n_jobs = 4,
        scoring = 'f1_micro'
    )

def report_model_scores(input_model, X_train, X_test, y_train, y_test):
    ### Function for printing out relevent classification model metrics.

    print("       --- Best Parameters ---")
    print(input_model.best_params_, '\n')

    print("Train Accuracy : ", accuracy_score(y_train, input_model.predict(X_train)))
    print("Test Accuracy : ", accuracy_score(y_test, input_model.predict(X_test)), '\n')

    print("     --- Classification Metrics (Test Set) ---")
    print(classification_report(y_true = y_test, y_pred = input_model.predict(X_test)))

## Adapted from the Sklearn Docs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          img_name='new_fig.png',
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./img/'+ str(img_name) , bbox_inches='tight')
    plt.close()
