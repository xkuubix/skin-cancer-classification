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
from pytorch_tabnet.tab_network import TabNet
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from Dataset import MappingHandler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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

    input_dim = X_train[0].shape[0]
    output_dim = 7
    group_attention_matrix = torch.randn(1, input_dim)

    model = TabNet(input_dim=input_dim, output_dim=output_dim,
                   group_attention_matrix=group_attention_matrix,
                   n_d=64, n_a=64, n_steps=5, gamma=1.5,
                   n_independent=2, n_shared=2,
                   epsilon=1e-15, momentum=0.02)
    lambda_sparse=1e-4



    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()


    optimizer = torch.optim.SGD(model.parameters(), lr=config['net_train']['lr'])
    criterion = torch.nn.CrossEntropyLoss(class_weights)

    best_val_loss = float('inf')
    patience = 1000
    counter = 0
    best_model_state_dict = None

    for epoch in range(config['net_train']['epochs']):
        model.train()

        for param in model.parameters():
            param.grad = None

        output, M_loss = model(X_train)
        output = torch.nn.functional.softmax(output, dim=1)
        loss = criterion(output, y_train)
        loss = loss - lambda_sparse * M_loss

        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)
        accuracy = (preds == y_train).float().mean().item() * 100
        balanced_acc = balanced_accuracy_score(y_train.cpu(), preds.cpu()) * 100

        model.eval()
        with torch.no_grad():
            val_output, val_M_loss = model(X_val)
            val_output = torch.nn.functional.softmax(val_output, dim=1)
            val_loss = criterion(val_output, y_val)
            val_loss = val_loss - lambda_sparse * val_M_loss

            val_preds = torch.argmax(val_output, dim=1)
            val_accuracy = (val_preds == y_val).float().mean().item() * 100
            val_balanced_acc = balanced_accuracy_score(y_val.cpu(), val_preds.cpu()) * 100

        if (epoch + 1) % 250 == 0:
            print(f"Epoch {epoch+1}/{config['net_train']['epochs']}: "
                  f"Train loss: {loss.item():.4f}, Train acc: {accuracy:.2f}%, "
                  f"Train balanced acc: {balanced_acc:.2f}%, "
                  f"Val loss: {val_loss.item():.4f}, Val acc: {val_accuracy:.2f}%, "
                  f"Val balanced acc: {val_balanced_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Epoch {epoch+1}/{config['net_train']['epochs']}: "
                  f"Train loss: {loss.item():.4f}, Train acc: {accuracy:.2f}%, "
                  f"Train balanced acc: {balanced_acc:.2f}%, "
                  f"Val loss: {val_loss.item():.4f}, Val acc: {val_accuracy:.2f}%, "
                  f"Val balanced acc: {val_balanced_acc:.2f}%")
            print("Early stopping triggered")
            break

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print(f"Best model found at epoch {epoch+1-counter}")

    model.eval()
    with torch.no_grad():
        test_output, test_M_loss = model(X_test)
        test_output = torch.nn.functional.softmax(test_output, dim=1)
        test_loss = criterion(test_output, y_test)
        test_loss = test_loss - lambda_sparse * test_M_loss

        test_preds = torch.argmax(test_output, dim=1)
        test_accuracy = (test_preds == y_test).float().mean().item() * 100
        test_balanced_acc = balanced_accuracy_score(y_test.cpu(), test_preds.cpu()) * 100

    print(f"Test loss: {test_loss.item():.4f}, Test acc: {test_accuracy:.2f}%, "
          f"Test balanced acc: {test_balanced_acc:.2f}%")

# %%
