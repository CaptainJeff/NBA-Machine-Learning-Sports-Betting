import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
DATABASE_PATH = "../../Data/dataset.sqlite"
DATASET = "dataset_2012-24"
MODEL_SAVE_PATH = '../../Models/XGBoost_{}%_UO-9.json'

# Configuration for XGBoost
XGB_PARAMS = {
    'max_depth': 20,
    'eta': 0.05,
    'objective': 'multi:softprob',
    'num_class': 3
}
EPOCHS = 750


def load_data(dataset):
    """Load data from SQLite database."""
    con = sqlite3.connect(DATABASE_PATH)
    try:
        data = pd.read_sql_query(
            f"select * from \"{dataset}\"", con, index_col="index")
    finally:
        con.close()
    return data


def preprocess_data(data):
    """Preprocess the data for model training."""
    OU = data['OU-Cover']
    total = data['OU']
    data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date',
              'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)
    data['OU'] = np.asarray(total)
    return data.values.astype(float), OU


def train_and_evaluate(data, labels):
    """Train the model and evaluate its performance."""
    acc_results = []
    for _ in tqdm(range(100)):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=.1)
        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test)

        model = xgb.train(XGB_PARAMS, train, EPOCHS)
        predictions = model.predict(test)
        predicted_labels = [np.argmax(z) for z in predictions]

        acc = round(accuracy_score(y_test, predicted_labels) * 100, 1)
        print(f"{acc}%")
        acc_results.append(acc)

        if acc == max(acc_results):
            model.save_model(MODEL_SAVE_PATH.format(acc))


# Main execution
data = load_data(DATASET)
processed_data, labels = preprocess_data(data)
train_and_evaluate(processed_data, labels)
