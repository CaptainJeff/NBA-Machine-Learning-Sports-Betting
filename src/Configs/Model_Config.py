XGBOOST_CONFIG = {"ml_model": {
    "target_column": 'Home-Team-Win',
    "drop_columns": ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
    "model_path_template": "../../Models/XGBoost_Models/XGBoost_{accuracy}%_ML-4.json",
    "filter_as_array_column": None
},
    "ou_model": {
    "target_column": 'OU-Cover',
    "drop_columns": ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date',
                     'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
    "model_path_template": "../../Models/XGBoost_Models/XGBoost_{accuracy}%_UO-9.json",
    "filter_as_array_column": "OU"
}
}
