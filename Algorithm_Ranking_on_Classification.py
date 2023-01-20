# Importing required modules and libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, HistGradientBoostingClassifier
import openml
openml.config.apikey = 'API Key'
from openml.runs import run_model_on_task
from openml.flows import get_flow
# from openml.datasets import get_dataset

task_ids = [18, 37, 59, 3954, 9983, 10101, 34539, 146065]
elements = []
df = pd.DataFrame(columns=['task_id', 'evaluation_metric', 'model', 'score'])

for task_id in task_ids:
    # Download the OpenMl task for task_id
    task = openml.tasks.get_task(task_id)

    Classifiers = [LogisticRegression(random_state=0, max_iter=5000), SGDClassifier(random_state=0),
                   KNeighborsClassifier(), DecisionTreeClassifier(random_state=0), GaussianNB(), SVC(random_state=0),
                   RandomForestClassifier(random_state=0), AdaBoostClassifier(random_state=0),
                   GradientBoostingClassifier(random_state=0), ExtraTreesClassifier(random_state=0),
                   HistGradientBoostingClassifier(random_state=0),
                   MLPClassifier(random_state=0, max_iter=5000)]

    evaluation_fn = ["predictive_accuracy", "area_under_roc_curve", "f_measure", "precision", "kappa"]
    models = ['LR', 'SGD', 'KNN', 'DTree', 'GaussNB', 'SVC', 'RForest', 'AdaBoost', 'GradBoost', 'ExtraTree',
              'HistGradBoost', 'MLP']

    for metric in evaluation_fn:
        print('\n', metric)
        for clf in Classifiers:
            # SimpleImputer handles missing values in data and replaces with specific values based on strategy chosen
            # StandardScaler rescales all features to mean=0 and variance=1
            numerical_pipeline = Pipeline(
                steps=[('Imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
            # Defining a scikit_learn pipeline/ workflow( a combination of transformer and an estimator )
            pipe = Pipeline([('numerical', numerical_pipeline), ('model', clf)])
            # Running the scikit_learn pipeline/workflow on the task using the function "run_model_on_task()"
            run_pipe = run_model_on_task(pipe, task, avoid_duplicate_runs=False, ).publish()
            flow = get_flow(run_pipe.flow_id)
            # Retrieving evaluations based on different evaluation functions (metrics)
            evaluation_score = openml.evaluations.list_evaluations(function=metric, flows=[flow.flow_id],
                                                                   output_format="dataframe")
            # print("{} - {} : {}".format(metric, clf, evaluation_score.mean(numeric_only=True)['value']))
            eval_values = evaluation_score.mean(numeric_only=True)['value']
            elements.append({
                'task_id': task_id,
                'evaluation_metric': metric,
                'model': clf,
                'score': eval_values
            })
            df = pd.DataFrame(elements, columns=['task_id', 'evaluation_metric', 'model', 'score'])
            print(df)

# Exporting the Dataframe into a CSV file
df.to_csv('Results.csv', sep=',', float_format='%.4f', header=True,
          columns=['task_id', 'evaluation_metric', 'model', 'score'], index=False)

