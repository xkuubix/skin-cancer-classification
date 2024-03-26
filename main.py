# %% IMPORTS AND SETTINGS
import yaml
import pickle
import torch
import numpy as np
import utils
import logging
from Dataset import HAM10000
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
# from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %% Load radiomics features
with open(config['dir']['pkl_train'], 'rb') as handle:
    train_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (train) from {config['dir']['pkl_train']}")
with open(config['dir']['pkl_val'], 'rb') as handle:
    val_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (val) from {config['dir']['pkl_val']}")
with open(config['dir']['pkl_test'], 'rb') as handle:
    test_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (test) from {config['dir']['pkl_test']}")
train_ds = HAM10000(df=train_df, mode='radiomics')
val_ds = HAM10000(df=val_df, mode='radiomics')
test_ds = HAM10000(df=test_df, mode='radiomics')
# %%
def map_labels(label):
    mapping = {
        "mel": 0,
        "nv": 1,
        "bcc": 2,
        "akiec": 3,
        "bkl": 4,
        "df": 5,
        "vasc": 6
    }
    return mapping[label.lower()]

cols_to_drop = ['lesion_id', 'image_id', 'img_path', 'seg_path', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset']
X_train = train_df.drop(columns=cols_to_drop)
y_train = train_df['dx'].map(map_labels)
X_val = val_df.drop(columns=cols_to_drop)
y_val = val_df['dx'].map(map_labels)
X_test = test_df.drop(columns=cols_to_drop)
y_test = test_df['dx'].map(map_labels)

if 'cuda' in config['device'] and torch.cuda.is_available():
    device = torch.device(config['device'])
else:
    device = torch.device('cpu')

# %% Feature selection
# import numpy as np
# import matplotlib.pyplot as plt

# alphas = np.linspace(0.00001,1,1000)
# lasso = Lasso(max_iter=10000)
# coefs = []
# scores = []
# scaler = StandardScaler()
# scaler.fit(X_train)
# for a in alphas:
#     lasso.set_params(alpha=a)
#     lasso.fit(scaler.transform(X_train), y_train)
#     # lasso.fit(X_train, y_train)
#     coefs.append(lasso.coef_)
#     scores.append(lasso.score(scaler.transform(X_train), y_train))
# ax = plt.gca()

# ax.plot(alphas, coefs)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlabel('alpha')
# plt.ylabel('Standardized Coefficients')
# plt.title('Lasso coefficients as a function of alpha');
# %%
scaler = StandardScaler()
scaler.fit(X_train)

pipeline = Pipeline([
    # ('scaler', StandardScaler()),
    ('model', Lasso(max_iter=10000))
])
param_grid = {
    'model__alpha': np.arange(1e-2, 1e-0, 1e-3),  # Range of alpha values to search
    # 'model__l1_ratio': np.arange(0.1, 1.0, 0.1)  # Range of l1_ratio values to search
}
search = GridSearchCV(pipeline,
                      param_grid,
                    #   n_jobs=-1,
                      cv=5, 
                      scoring="neg_mean_squared_error",
                      verbose=0)

search.fit(scaler.transform(X_train), y_train)
print("Best Parameters:", search.best_params_)
print("Best Score:", search.best_score_)
best_model = search.best_estimator_
feature_importances = best_model.named_steps['model'].coef_
# %%
print(f'Non-zero feature coefs: {np.count_nonzero(feature_importances)}')
model = Lasso(alpha=search.best_params_['model__alpha'],
              max_iter=10000)
model.fit(scaler.transform(X_train), y_train)

selected_features = X_train.columns[np.abs(feature_importances) > 0]
X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
X_test_selected = X_test[selected_features]



X_train_selected = np.vstack(X_train_selected.values).astype(float)
X_val_selected = np.vstack(X_val_selected.values).astype(float)
X_test_selected = np.vstack(X_test_selected.values).astype(float)

y_train_v = y_train.values.astype(int)
y_val_v = y_val.values.astype(int)
y_test_v = y_test.values.astype(int)


# %% Define objective function for hyperparameter search
def objective(trial):
    # Define hyperparameters to search
    n_d = trial.suggest_int('n_d', 8, 64)
    n_a = trial.suggest_int('n_a', 8, 64)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    n_independent = trial.suggest_int('n_independent', 1, 5)
    n_shared = trial.suggest_int('n_shared', 1, 5)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    lambda_sparse = trial.suggest_float('lambda_sparse', 0., 0.1)
    
    # Define optimizer parameters to search
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'SGD'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    # Initialize optimizer based on the chosen name
    if optimizer_name == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif optimizer_name == 'SGD':
        optimizer_fn = torch.optim.SGD
    # Initialize optimizer parameters
    optimizer_params = {
        'lr': learning_rate,
        'weight_decay': weight_decay
    }

    # Initialize TabNet model with hyperparameters
    model = TabNetClassifier(n_d=n_d,
                             n_a=n_a,
                             n_steps=n_steps,
                             n_independent=n_independent,
                             n_shared=n_shared,
                             gamma=gamma,
                             lambda_sparse=lambda_sparse,
                             verbose=0,
                             optimizer_fn=optimizer_fn,
                             optimizer_params=optimizer_params)
    
    # Train the model
    model.fit(
        X_train=X_train_selected,
        y_train=y_train_v,
        eval_set=[(X_train_selected, y_train_v),
                   (X_val_selected, y_val_v)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy', 'logloss'],
        max_epochs=1000,
        patience=100)
    
    # Evaluate the model
    loss = np.min(model.history['valid_logloss'])
    
    return loss


import warnings
warnings.filterwarnings('ignore')

# Perform hyperparameter search
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# Get best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

#%%
# from matplotlib import pyplot as plt
# from sklearn import metrics
# predicted = classifier.predict_proba(X_test_v)
# predicted_labels = predicted.argmax(axis=1)
# confusion_matrix = metrics.confusion_matrix(y_test_v, predicted_labels)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'])
# cm_display.plot()
# plt.show()