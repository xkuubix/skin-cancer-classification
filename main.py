# %% IMPORTS AND SETTINGS
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
import utils
import logging
from Dataset import HAM10000
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.augmentations import ClassificationSMOTE
# from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

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
# %%
# def pickleLoader(pklFile):
#     i = 0
#     try:
#         while True:
#             i += 1
#             print(f'loading i={i}')
#             yield pickle.load(pklFile)
#     except EOFError:
#         pass
# %% Load radiomics features
with open(config['dir']['pkl_train'], 'rb') as handle:
    # train_df = pd.DataFrame([])
    # for event in pickleLoader(handle):
    #     train_df = pd.concat([train_df, event])
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
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6
    }
    return mapping[label.lower()]

unique_localizations = np.concatenate((train_df['localization'].unique(), val_df['localization'].unique(), test_df['localization'].unique()))
unique_localizations = np.unique(unique_localizations)

def map_localizations(localization):
    mapping = {}
    for i, loc in enumerate(unique_localizations):
        mapping[loc] = i
    return mapping[localization.lower()]

def map_sex(sex):
    mapping = {
        "male": 0,
        "female": 1,
        "unknown": 2
    }
    return mapping[sex.lower()]

# %%
# cols_to_drop = ['lesion_id', 'image_id', 'img_path', 'seg_path', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset']
cols_to_drop = ['lesion_id', 'image_id', 'img_path', 'seg_path', 'dx', 'dx_type', 'dataset']

X_train = train_df.drop(columns=cols_to_drop)
X_val = val_df.drop(columns=cols_to_drop)
X_test = test_df.drop(columns=cols_to_drop)


# Mapping for labels
y_train = train_df['dx'].map(map_labels)
y_val = val_df['dx'].map(map_labels)
y_test = test_df['dx'].map(map_labels)

# Mapping for localization
if 'localization' in X_train.columns:
    X_train['localization'] = X_train['localization'].map(map_localizations)
    X_val['localization'] = X_val['localization'].map(map_localizations)
    X_test['localization'] = X_test['localization'].map(map_localizations)

# Mapping for sex
if 'sex' in X_train.columns:
    X_train['sex'] = X_train['sex'].map(map_sex)
    X_val['sex'] = X_val['sex'].map(map_sex)
    X_test['sex'] = X_test['sex'].map(map_sex)

# Mapping for age
if 'age' in X_train.columns:
    mean_age = 5 * round(X_train['age'].mean() / 5)
    X_train['age'] = X_train['age'].fillna(mean_age)
    X_val['age'] = X_val['age'].fillna(mean_age)
    X_test['age'] = X_test['age'].fillna(mean_age)
    X_train['age'] = X_train['age'].astype(int)
    X_val['age'] = X_val['age'].astype(int)
    X_test['age'] = X_test['age'].astype(int)



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
# scaler = StandardScaler()
# scaler.fit(X_train)

# pipeline = Pipeline([
#     # ('scaler', StandardScaler()),
#     ('model', Lasso(max_iter=10000))
# ])
# param_grid = {
#     'model__alpha': np.arange(1e-4, 1e-0, 1e-3),  # Range of alpha values to search
#     # 'model__l1_ratio': np.arange(0.1, 1.0, 0.1)  # Range of l1_ratio values to search
# }
# search = GridSearchCV(pipeline,
#                       param_grid,
#                     #   n_jobs=-1,
#                       cv=2, 
#                       scoring="neg_mean_squared_error",
#                       verbose=0)

# search.fit(scaler.transform(X_train), y_train)
# print("Best Parameters:", search.best_params_)
# print("Best Score:", search.best_score_)
# best_model = search.best_estimator_
# feature_importances = best_model.named_steps['model'].coef_


# %%
# alpha = 0.0061
# alpha = search.best_params_['model__alpha'] # 0.0061
# alpha = 0.05
# model = Lasso(alpha=alpha,
#               max_iter=10000)
# scaler = StandardScaler()
# scaler.fit(X_train)
# model.fit(scaler.transform(X_train), y_train)
# print(f'Non-zero feature coefs: {np.count_nonzero(model.coef_)}')
# selected_features = X_train.columns[np.abs(model.coef_) > 0]
# X_train_selected = X_train[selected_features]
# X_val_selected = X_val[selected_features]
# X_test_selected = X_test[selected_features]


# %%
X_train_selected = np.vstack(X_train.values).astype(float)
X_val_selected = np.vstack(X_val.values).astype(float)
X_test_selected = np.vstack(X_test.values).astype(float)

# X_train_selected = np.vstack(X_train_selected.values).astype(float)
# X_val_selected = np.vstack(X_val_selected.values).astype(float)
# X_test_selected = np.vstack(X_test_selected.values).astype(float)

scaler = StandardScaler()
scaler.fit(X_train_selected)
X_train_selected = scaler.transform(X_train_selected)
X_val_selected = scaler.transform(X_val_selected)
X_test_selected = scaler.transform(X_test_selected)

y_train_v = y_train.values.astype(int)
y_val_v = y_val.values.astype(int)
y_test_v = y_test.values.astype(int)

# class_weights = compute_class_weight(class_weight="balanced",
#                                      classes=np.unique(y_train_v),
#                                      y=y_train_v)



# class_weights = [   
#                   1.0,  # akiec
#                   1.0,  # bcc
#                   1.0,  # bkl
#                   1.0,  # df
#                   1.0,  # mel
#                    .1,  # nv
#                   1.0,  # vasc
#                 ]
# %%
# sklearn Gradient Boosting Classifier
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(max_iter=50000,
                                       learning_rate=0.01,
                                       l2_regularization=0.001,
                                    #    class_weight='balanced',
                                       validation_fraction=None,
                                       early_stopping=False,
                                    #    max_depth=None,
                                    #    max_leaf_nodes=None,
                                       verbose=1,
                                       random_state=seed)
model.fit(X_train_selected, y_train_v)
predicted = model.predict(X_test_selected)
confusion_matrix = metrics.confusion_matrix(y_test_v, predicted)
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
cm_display.plot()
plt.show()
# Generate classification report
report = classification_report(y_test_v, predicted)
print(report)

# %% Self-supervised pretraining
n_d = 128
n_a = 128
n_steps = 4
n_independent = 2
n_shared = 2
gamma = 1.3
lambda_sparse = 1e-2
# %%
pretrainer = TabNetPretrainer(
    n_d=n_d,
    n_a=n_a,
    n_steps=n_steps,
    n_independent=n_independent,
    n_shared=n_shared,
    gamma=gamma,
    lambda_sparse=lambda_sparse,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-4, weight_decay=1e-2),
    verbose=0,
    mask_type='entmax',
    device_name=device
)
pretrainer.fit(
    X_train=X_train_selected,
    eval_set=[X_val_selected],
    max_epochs=3000,
    patience=200,
    batch_size=64, #lower batch is better for pretraining
    virtual_batch_size=32,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.33,
)
# %%
model = TabNetClassifier(
    n_d=n_d,
    n_a=n_a,
    n_steps=n_steps,
    n_independent=n_independent,
    n_shared=n_shared,
    gamma=gamma,
    lambda_sparse=lambda_sparse,
    verbose=0,
    mask_type='entmax',
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-4, weight_decay=1e-2),
    # scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    # scheduler_params=dict(mode='min', factor=0.1, patience=20, min_lr=1e-6),
    device_name=device,
    # grouped_features=[[0, 1, 2]],
    # momentum=0.3,
)
model.fit(
    X_train=X_train_selected,
    y_train=y_train_v,
    eval_set=[(X_train_selected, y_train_v),
              (X_val_selected, y_val_v)],
    eval_name=['train', 'valid'],
    eval_metric=['balanced_accuracy', 'accuracy', 'logloss'],
    max_epochs=10000,
    patience=200,
    batch_size=512,
    virtual_batch_size=256,
    augmentations=ClassificationSMOTE(device_name=device, seed=config['seed']),
    # weights=1,
    loss_fn=torch.nn.CrossEntropyLoss(), #weight=torch.tensor(class_weights, dtype=torch.float).to(device)),
    # from_unsupervised=pretrainer,
    # warm_start=True
    )
plt.plot(model.history['train_accuracy'], label='train')
plt.plot(model.history['valid_accuracy'], label='valid')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.plot(model.history['train_balanced_accuracy'], label='train')
plt.plot(model.history['valid_balanced_accuracy'], label='valid')
plt.title('balanced_accuracy')
plt.legend()
plt.show()

plt.plot(model.history['train_logloss'], label='train')
plt.plot(model.history['valid_logloss'], label='valid')
plt.title('logloss')
plt.legend()
plt.show()

predicted = model.predict_proba(X_test_selected)
predicted_labels = predicted.argmax(axis=1)
confusion_matrix = metrics.confusion_matrix(y_test_v, predicted_labels)
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
cm_display.plot()
plt.show()
# Make predictions on the test set
predicted = model.predict(X_test_selected)

# Generate classification report
report = classification_report(y_test_v, predicted)
print(report)
# %% Define objective function for hyperparameter search
def objective(trial):
    # Define hyperparameters to search
    # n_d = trial.suggest_int('n_d', 8, 64)
    # n_a = trial.suggest_int('n_a', 8, 64)
    # n_steps = trial.suggest_int('n_steps', 3, 10)
    n_d = 64
    n_a = 64
    n_steps = 3
    # n_independent = trial.suggest_int('n_independent', 1, 5)
    # n_shared = trial.suggest_int('n_shared', 1, 5)
    # gamma = trial.suggest_float('gamma', 1.0, 2.0)
    # lambda_sparse = trial.suggest_float('lambda_sparse', 0., 0.1)
    
    # Define optimizer parameters to search
    # optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'SGD'])
    optimizer_name = 'Adam'
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    # learning_rate = 1e-2
    # weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    weight_decay = 1e-3
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
                            #  n_independent=n_independent,
                            #  n_shared=n_shared,
                            #  gamma=gamma,
                            #  lambda_sparse=lambda_sparse,
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
        max_epochs=5000,
        patience=500,
        loss_fn=torch.nn.CrossEntropyLoss()
            #weight=torch.tensor(class_weights, dtype=torch.float).to(device)))
    )
    # Evaluate the model
    loss = np.min(model.history['valid_logloss'])
    
    return loss

# Perform hyperparameter search
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
# {'n_d': 60, 'n_a': 15, 'n_steps': 8, 'n_independent': 5, 'n_shared': 5}
#%%
# from matplotlib import pyplot as plt
# from sklearn import metrics
# predicted = classifier.predict_proba(X_test_v)
# predicted_labels = predicted.argmax(axis=1)
# confusion_matrix = metrics.confusion_matrix(y_test_v, predicted_labels)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc'])
# cm_display.plot()
# plt.show()
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
kernel = cv2.getStructuringElement(1,(17,17)) # Kernel for the morphological filtering
paths = train_df['img_path'].values
for i in range(5):
    idx=np.random.randint(1,len(paths))
    src = cv2.imread(paths[idx])
    #print(src.shape)
    grayScale = cv2.cvtColor( src, cv2.COLOR_BGR2GRAY ) #1 Convert the original image to grayscale
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) #2 Perform the blackHat filtering on the grayscale image to find the hair countours
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY) # intensify the hair countours in preparation for the inpainting algorithm
    dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA) # inpaint the original image depending on the mask
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    plt.figure(figsize=(20,10))
    plt.subplot(1,5,1).set_title('Original')
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.subplot(1,5,2).set_title('Final')
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.subplot(1,5,3).set_title('Grayscale')
    plt.imshow(cv2.cvtColor(grayScale, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.subplot(1,5,4).set_title('Blackhat')
    plt.imshow(cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.subplot(1,5,5).set_title('Thresh2')
    plt.imshow(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB), interpolation='nearest')

# %%
# Define the number of folds
n_splits = 5
from sklearn.model_selection import StratifiedKFold

# Initialize the StratifiedKFold object
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

# Initialize lists to store the evaluation metrics
train_accs = []
valid_accs = []
train_balanced_accs = []
valid_balanced_accs = []
train_losses = []
valid_losses = []

test_accs = []
test_balanced_accs = []

# Perform k-fold cross validation
X_train_selected = np.vstack((X_train_selected, X_val_selected))
y_train_v = np.concatenate((y_train_v, y_val_v))
for fold, (train_index, valid_index) in enumerate(skf.split(X_train_selected, y_train_v)):
    print(f"Fold {fold+1}/{n_splits}")
    
    # Split the data into training and validation sets

    X_train_fold, X_valid_fold = X_train_selected[train_index], X_train_selected[valid_index]
    y_train_fold, y_valid_fold = y_train_v[train_index], y_train_v[valid_index]
    
    # Initialize a new TabNetClassifier model
    pretrainer = TabNetPretrainer(
    n_d=n_d,
    n_a=n_a,
    n_steps=n_steps,
    n_independent=n_independent,
    n_shared=n_shared,
    gamma=gamma,
    lambda_sparse=lambda_sparse,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-4, weight_decay=1e-2),
    verbose=0,
    mask_type='entmax',
    device_name=device
    )
    pretrainer.fit(
        X_train=X_train_fold,
        eval_set=[X_valid_fold],
        max_epochs=3000,
        patience=200,
        batch_size=64, #lower batch is better for pretraining
        virtual_batch_size=32,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.33,
    )
    model = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        n_independent=n_independent,
        n_shared=n_shared,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        verbose=0,
        mask_type='entmax',
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-4, weight_decay=1e-2),
    )
    
    # Fit the model on the training data
    model.fit(
        X_train=X_train_fold,
        y_train=y_train_fold,
        eval_set=[(X_train_fold, y_train_fold), (X_valid_fold, y_valid_fold)],
        eval_name=['train', 'valid'],
        eval_metric=['balanced_accuracy', 'accuracy', 'logloss'],
        max_epochs=10000,
        patience=200,
        batch_size=512,
        virtual_batch_size=256,
        augmentations=ClassificationSMOTE(device_name=device, seed=seed),
        from_unsupervised=pretrainer
    )
    
    # Evaluate the model on the training and validation data
    best_loss_epoch = np.argmin(model.history['valid_logloss'])
    train_acc = model.history['train_accuracy'][best_loss_epoch]
    valid_acc = model.history['valid_accuracy'][best_loss_epoch]
    train_balanced_acc = model.history['train_balanced_accuracy'][best_loss_epoch]
    valid_balanced_acc = model.history['valid_balanced_accuracy'][best_loss_epoch]
    train_loss = model.history['train_logloss'][best_loss_epoch]
    valid_loss = model.history['valid_logloss'][best_loss_epoch]
    
    # Append the evaluation metrics to the lists
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    train_balanced_accs.append(train_balanced_acc)
    valid_balanced_accs.append(valid_balanced_acc)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    predicted = model.predict_proba(X_test_selected)
    predicted_labels = predicted.argmax(axis=1)
    test_acc = metrics.accuracy_score(y_test_v, predicted_labels)
    test_balanced_acc = metrics.balanced_accuracy_score(y_test_v, predicted_labels)
    
    test_accs.append(test_acc)
    test_balanced_accs.append(test_balanced_acc)

# Print the average evaluation metrics across all folds
print(f"Average Train Accuracy: {np.mean(train_accs):.4f}")
print(f"Average Valid Accuracy: {np.mean(valid_accs):.4f}")
print(f"Average Train Balanced Accuracy: {np.mean(train_balanced_accs):.4f}")
print(f"Average Valid Balanced Accuracy: {np.mean(valid_balanced_accs):.4f}")
print(f"Average Train Loss: {np.mean(train_losses):.4f}")
print(f"Average Valid Loss: {np.mean(valid_losses):.4f}")
print(f"Average Test Accuracy: {np.mean(test_accs):.4f}")
print(f"Average Test Balanced Accuracy: {np.mean(test_balanced_accs):.4f}")
