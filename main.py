 # %% IMPORTS AND SETTINGS
import pandas as pd
import yaml
import utils
import pickle
import logging
import torch
import numpy as np
import albumentations as A
from Dataset import HAM10000
# from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.utils.class_weight import compute_class_weight
from Dataset import MappingHandler
from pytorch_tabnet.augmentations import ClassificationSMOTE
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
from utils import print_metrics
from sklearn.model_selection import StratifiedKFold

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
device = config['device'] if torch.cuda.is_available() else 'cpu'

# %%

# Dataset
with open(config['dir']['pkl_train'], 'rb') as handle:
    train_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (train) from {config['dir']['pkl_train']}")
with open(config['dir']['pkl_val'], 'rb') as handle:
    val_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (val) from {config['dir']['pkl_val']}")
with open(config['dir']['pkl_test'], 'rb') as handle:
    test_df = pickle.load(handle)
    logger.info(f"Loaded radiomics features (test) from {config['dir']['pkl_test']}")


img_size = config['dataset']['img_size']
transforms_train = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])

transforms_val_test = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])


# %% Tabnet
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

cls_reports = []
fold_results = []
test_results = []
k = config['dataset']['k_folds']
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
df = pd.concat([train_df, val_df], ignore_index=True)
kf.get_n_splits(df)
mapping_handler = MappingHandler()
# %%
for fold_index, (train_indices, val_indices) in enumerate(kf.split(df, df['dx'])):
    train_fold = df.iloc[train_indices]
    val_fold = df.iloc[val_indices]

    combined = pd.concat([train_fold.copy(), val_fold.copy()], ignore_index=True)
    combined_features = combined[combined.columns[10:]]
    combined_features = combined_features.applymap(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
    columns_to_keep = combined_features.T.drop_duplicates().T.columns
    columns_to_keep = combined.columns[:10].tolist() + columns_to_keep.tolist()
    train_fold = train_fold[columns_to_keep]
    val_fold = val_fold[columns_to_keep]
    test_fold = test_df[columns_to_keep]

    train_ds = HAM10000(df=train_fold, transform=transforms_train, mode=config['dataset']['mode'])
    val_ds = HAM10000(df=val_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    test_ds = HAM10000(df=test_fold, transform=transforms_val_test, mode=config['dataset']['mode'])

    X_train = np.vstack([x['features'] for x in train_ds])
    y_train = np.array([x['label'] for x in train_ds])
    X_val = np.vstack([x['features'] for x in val_ds])
    y_val = np.array([x['label'] for x in val_ds])
    X_test = np.vstack([x['features'] for x in test_ds])
    y_test = np.array([x['label'] for x in test_ds])


    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}


    # cat_idxs = []
    # cat_dims = []

    # unsupervised_model = TabNetPretrainer(
    #     cat_idxs=cat_idxs,
    #     cat_dims=cat_dims,
    #     cat_emb_dim=3,
    #     optimizer_fn=torch.optim.Adam,
    #     optimizer_params=dict(lr=2e-4),
    #     mask_type='entmax',  # "sparsemax",
    #     n_shared_decoder=2,  # nb shared glu for decoding
    #     n_indep_decoder=2,  # nb independent glu for decoding
    #     verbose=5,
    #     device_name='cpu'
    # )
    # unsupervised_model.fit(
    #     X_train=X_train,
    #     eval_set=[X_val],
    #     max_epochs=config['net_train']['epochs'],
    #     patience=150, batch_size=2048,
    #     virtual_batch_size=128, num_workers=0,
    #     drop_last=False,
    #     pretraining_ratio=0.5,
    # )



    model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2,
        epsilon=1e-15, momentum=0.02,
        clip_value=2., lambda_sparse=1e-4,
        seed=seed, verbose=1,
        device_name='cpu',
        # optimizer_fn=torch.optim.SGD,
    )



    aug = ClassificationSMOTE(device_name='cpu', p=0.2, seed=seed)

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['accuracy', 'balanced_accuracy', 'logloss'],
        max_epochs=config['net_train']['epochs'],
        patience=20, batch_size=config['net_train']['batch_size'],
        virtual_batch_size=128, num_workers=config['net_train']['num_workers'],
        drop_last=False,
        weights=class_weights_dict,
        augmentations=aug,
        # from_unsupervised=unsupervised_model
    )

    val_preds = model.predict(X_val)
    accuracy = (val_preds == y_val).mean() * 100
    fold_results.append(accuracy)
    logger.info(f"Fold {fold_index + 1}, Validation Accuracy: {accuracy:.2f}%")

    test_preds = model.predict(X_test)
    test_accuracy = (test_preds == y_test).mean() * 100
    test_results.append(test_accuracy)
    report = classification_report(y_test, test_preds, target_names=mapping_handler.mapping.keys(), output_dict=True)
    probabilities = model.predict_proba(X_test)
    true_targets = y_test
    roc_all = roc_auc_score(true_targets, probabilities,
                            multi_class='ovr', average=None)
    
    roc_micro = roc_auc_score(true_targets, probabilities,
                              multi_class='ovr', average="micro")
    for item in range(len(mapping_handler.mapping.keys())):
        label = list(mapping_handler.mapping.keys())[item]
        report[label]['roc_auc'] = roc_all[item]

    report['weighted avg']["roc_auc"] = roc_auc_score(true_targets, probabilities,
                                                      multi_class='ovr', average="weighted")
    report['macro avg']['roc_auc'] = roc_auc_score(true_targets, probabilities,
                                                   multi_class='ovr', average="macro")
    report['roc_auc_micro'] = roc_micro
    report['balanced_accuracy'] = balanced_accuracy_score(true_targets, test_preds)

    
    cls_reports.append(report)
    logger.info(f"Fold {fold_index + 1}, Test Accuracy: {test_accuracy:.2f}%")

print(f"Mean Validation Accuracy: {np.mean(fold_results):.4f}")
print(f"Mean Test Accuracy: {np.mean(test_results):.4f}")

print_metrics(cls_reports)
# %%
