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
from pytorch_tabnet.tab_network import TabNet
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from Dataset import MappingHandler
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
    class_weights = torch.from_numpy(class_weights).float()
    # class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    input_dim = X_train[0].shape[0]
    output_dim = 7
    group_attention_matrix = torch.randn(output_dim, input_dim)

    model = TabNet(input_dim=input_dim, output_dim=output_dim,
                   group_attention_matrix=group_attention_matrix,
                   n_d=64, n_a=64, n_steps=5, gamma=1.5,
                   n_independent=2, n_shared=2,
                   epsilon=1e-15, momentum=0.02)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()


    # smote = SMOTE(random_state=seed)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # X_train = torch.from_numpy(X_train_resampled).float()
    # y_train = torch.from_numpy(y_train_resampled).long()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['net_train']['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config['net_train']['epochs']):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)
        
        preds = torch.argmax(outputs[0], dim=1)
        accuracy = (preds == y_train).float().mean().item() * 100
        balanced_acc = balanced_accuracy_score(y_train.cpu(), preds.cpu()) * 100

        loss = criterion(outputs[0], y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs[0], y_val)
            
            val_preds = torch.argmax(val_outputs[0], dim=1)
            val_accuracy = (val_preds == y_val).float().mean().item() * 100
            val_balanced_acc = balanced_accuracy_score(y_val.cpu(), val_preds.cpu()) * 100

        print(f"Epoch {epoch+1}/{config['net_train']['epochs']}: "
                f"Train loss: {loss.item():.4f}, Train acc: {accuracy:.2f}%, "
                f"Train balanced acc: {balanced_acc:.2f}%, "
                f"Val loss: {val_loss.item():.4f}, Val acc: {val_accuracy:.2f}%, "
                f"Val balanced acc: {val_balanced_acc:.2f}%")

# %%
