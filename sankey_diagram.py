# %% IMPORTS AND SETTINGS
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
import utils
from net_utils import get_proba
import logging
import albumentations as A
from Dataset import HAM10000, MappingHandler
from sklearn.model_selection import StratifiedKFold  #, KFold
from model import DeepRadiomicsClassifier, RadiomicsClassifier, ImageClassifier
import random
import seaborn as sns
import matplotlib.pyplot as plt

# with hair
# paths = ['model_fold_0_f667fabfa1c747bc85e49b1e0b7dfe3e',
#          'model_fold_1_dbca0643b5054fe9a7d77ac2f0dcb86e',
#          'model_fold_2_6fdc181b2f5d4772a3a9b5f43b75b7d5',
#          'model_fold_3_73591c86886046a3af9877610bcf56ff',
#          'model_fold_4_dc73e29eb99f4297bfb20d116f3a7e6a'
        #  ]
# no hair
paths = ['model_fold_0_e337ad33562b4c61aaa956ef1f07befa',
         'model_fold_1_9dff589f89c743448e1405d58daf5c38',
         'model_fold_2_f58823fc62934ce7b8659ea4c85cbb59',
         'model_fold_3_45d4591cd68d453e8ddbbdd4b050dd3a',
         'model_fold_4_743f42ba09ce44ef8b8db3cfc533a485'
         ]

load = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# DATASET --------------------------------
img_size = config['dataset']['img_size']
crop_size = 256
transforms_train = A.Compose([
    A.RandomResizedCrop(crop_size, crop_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.GridDistortion(distort_limit=0.1, p=0.5),
    A.ColorJitter(p=0.5),
    A.GaussNoise(p=0.5),
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])

transforms_val_test = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
    ])

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
# %%
device = config['device'] if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Mode: {config['dataset']['mode']}")
if config['dataset']['mode'] in ['hybrid', 'images']:
    print(f"Backbone: {config['net_train']['backbone']}")

num_classes = 1 if config['net_train']['criterion'] == 'bce' else len(train_df['dx'].unique())
if config['net_train']['criterion'] == 'bce':
    criterion = torch.nn.BCELoss()
elif config['net_train']['criterion'] == 'ce':
    criterion = torch.nn.CrossEntropyLoss()

# %% ---------------------K-Fold Cross Validation---------------------
fold_results = {}
k = config['dataset']['k_folds']
# kf = KFold(n_splits=k, shuffle=True, random_state=seed)
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
# Concatenate train and validation set for k-fold cross validation
df = pd.concat([train_df, val_df], ignore_index=True)
kf.get_n_splits(df)
for fold_index, (train_indices, val_indices) in enumerate(kf.split(df, df['dx'])):
    if load:
        break
    train_fold = df.iloc[train_indices]
    val_fold = df.iloc[val_indices]
    print('--------------------------------')
    print(f"Fold {fold_index}:")
    print(f"  Train: len={len(train_indices)}")
    print(f"  Train class distribution: {df.iloc[train_indices]['dx'].value_counts(normalize=True)}")
    print(f"  Val:  len={len(val_indices)}")
    print(f"  Validation class distribution: {df.iloc[val_indices]['dx'].value_counts(normalize=True)}")
    # feature selecct + normalization (copy df cuz it is in the loop)
    train_fold, val_fold, test_fold, _ = utils.prepare_data_for_fold(
        train_fold.copy(), val_fold.copy(), test_df.copy(), random_state=seed)

    print(f"Training mean (column 11): {train_fold.iloc[:, 10].mean()}")
    print(f"Validation mean (column 11): {val_fold.iloc[:, 10].mean()}")
    print(f"Test mean (column 11): {test_fold.iloc[:, 10].mean()}")

    # datasets & dataloaders
    train_ds = HAM10000(df=train_fold, transform=transforms_train, mode=config['dataset']['mode'])
    val_ds = HAM10000(df=val_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    test_ds = HAM10000(df=test_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    
    mapping_handler = MappingHandler().mapping
    labels = [int(mapping_handler[label.upper()]) for label in train_fold['dx'].values]
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config['net_train']['batch_size'],
        sampler=utils.generate_sampler(labels),
        pin_memory=True, num_workers=config['net_train']['num_workers'])
    
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=config['net_train']['batch_size'],
        shuffle=False, pin_memory=True, num_workers=config['net_train']['num_workers'])

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config['net_train']['batch_size'],
        shuffle=False, pin_memory=True, num_workers=config['net_train']['num_workers'])
    
    # retrieve model state dict
    path = config['dir']['models'] + paths[fold_index] + '.pth'
    state_dict = torch.load(path)
    
    # create model instance
    if config['dataset']['mode'] == 'hybrid':
        feature_size = len(state_dict['radiomic_norm.weight'])
        model = DeepRadiomicsClassifier(radiomic_feature_size=feature_size,
                                        num_classes=num_classes,
                                        backbone=config['net_train']['backbone'])
    elif config['dataset']['mode'] == 'radiomics':
        feature_size = len(state_dict['radiomic_norm.weight'])
        model = RadiomicsClassifier(radiomic_feature_size=feature_size,
                                    num_classes=num_classes)
    elif config['dataset']['mode'] == 'images':
        model = ImageClassifier(num_classes=num_classes,
                                backbone=config['net_train']['backbone'])

    # load parameters
    model.load_state_dict(state_dict)
    model.to(device=device)
    probabilities = get_proba(model, test_dl, config, device)
    fold_results[fold_index] = probabilities
    break # only one fold for now

# %%
# Save results
def compute_entropy(predictions):
    predictions = np.clip(predictions, a_min=1e-10, a_max=1.0)
    entropies = -np.sum(predictions * np.log(predictions), axis=1)
    return entropies

def average_entropy(predictions):
    entropies = compute_entropy(predictions)
    return np.mean(entropies)

# %%
if not load:
    # Compute average entropy
    avg_entropy = average_entropy(fold_results[0])
    print(f"Average Entropy across all samples: {avg_entropy:.4f}")
    # Fold 1:
    # hair Average Entropy across all samples: 0.2902
    # no-hair Average Entropy across all samples: 0.3177
    save_path = config['dir']['results'] + 'hair/fold_results_no_hair.pkl'
    with open(save_path, 'wb') as handle:
        pickle.dump(fold_results, handle)
        logger.info(f"Saved fold results to {config['dir']['results']}fold_results.pkl")
elif load:
    load_path_1 = config['dir']['results'] + 'hair/fold_results_hair.pkl'
    load_path_2 = config['dir']['results'] + 'no_hair/fold_results_no_hair.pkl'
    with open(load_path_1, 'rb') as handle:
        hair_results = pickle.load(handle)
        logger.info(f"Loaded fold results from {config['dir']['results']}fold_results.pkl")
    with open(load_path_2, 'rb') as handle:
        no_hair_results = pickle.load(handle)
        logger.info(f"Loaded fold results from {config['dir']['results']}fold_results.pkl")
# %%
# Create a DataFrame of predicted labels and entropy
predicted_labels_hair = np.argmax(hair_results[0], axis=1)
entropies_hair = compute_entropy(hair_results[0])


predicted_labels_no_hair = np.argmax(no_hair_results[0], axis=1)
entropies_no_hair = compute_entropy(no_hair_results[0])

mapping_handler = MappingHandler()
predicted_labels_hair = [mapping_handler._convert(label) for label in predicted_labels_hair]
predicted_labels_no_hair = [mapping_handler._convert(label) for label in predicted_labels_no_hair]
predicted_labels_hair = [label.lower() for label in predicted_labels_hair]
predicted_labels_no_hair = [label.lower() for label in predicted_labels_no_hair]


# predicted_labels_hair = test_df['dx'].values
# predicted_labels_no_hair = test_df['dx'].values

boxplot_df = pd.DataFrame({
    'predicted_label_hair': predicted_labels_hair,
    'entropy_hair': entropies_hair,
    'predicted_label_no_hair': predicted_labels_no_hair,
    'entropy_no_hair': entropies_no_hair
})

melted_df = pd.melt(boxplot_df, id_vars=['predicted_label_hair', 'predicted_label_no_hair'], 
                    value_vars=['entropy_hair', 'entropy_no_hair'],
                    var_name='condition', value_name='entropy')

melted_df['predicted_class'] = melted_df.apply(
    lambda row: row['predicted_label_hair'] if row['condition'] == 'entropy_hair' else row['predicted_label_no_hair'], axis=1)

# Plot the boxplot for both hair and no hair predictions
plt.figure(figsize=(8, 6))
sns.boxplot(x='predicted_class', y='entropy', hue='condition', data=melted_df)
plt.xlabel('Predicted Class')
plt.ylabel('Entropy')
plt.legend(title='condition', loc='best')
plt.title('Entropy Distribution by Predicted Class (with and without hair)', fontsize=14)
# plt.title('Entropy Distribution by Ground Truth (with and without hair)', fontsize=14)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.grid(axis='y', linestyle=':', alpha=0.8)
plt.gca().yaxis.set_ticks_position('none')
plt.gca().xaxis.set_ticks_position('none')
plt.xlabel('')
plt.ylabel('')
plt.gca().tick_params(axis='both', which='major', labelsize=12)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], ['Hair', 'No Hair'], loc='upper right')
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
plt.show()
# %%
hair_predictions = np.argmax(hair_results[0], axis=1) + 7
no_hair_predictions = np.argmax(no_hair_results[0], axis=1) + 14
target = test_df['dx'].values
mapping_handler = MappingHandler().mapping
target = [int(mapping_handler[label.upper()]) for label in target]
print(f"Hair predictions: {hair_predictions}")
print(f"No hair predictions: {no_hair_predictions}")
print(f"Target: {target}")
# %%
import plotly.graph_objects as go
labels = ['akiec gt', 'bcc gt', 'bkl gt', 'df gt', 'mel gt', 'nv gt', 'vasc gt']
labels = labels + ['akiec h', 'bcc h', 'bkl h', 'df h', 'mel h', 'nv h', 'vasc h']
labels = labels + ['akiec nh', 'bcc nh', 'bkl nh', 'df nh', 'mel nh', 'nv nh', 'vasc nh']

df = pd.DataFrame({
    'Ground Truth': target,
    'Prediction 1': hair_predictions,
    'Prediction 2': no_hair_predictions
})

contingency_table = pd.crosstab([df['Ground Truth'], df['Prediction 1']], df['Prediction 2'])
print(contingency_table)
total_samples = contingency_table.sum().sum()
normalized_contingency_table = contingency_table / total_samples
print(normalized_contingency_table)

nodes = list(np.unique(df[['Ground Truth', 'Prediction 1', 'Prediction 2']].values))
node_mapping = {label: idx for idx, label in enumerate(nodes)}
links = []
colors = ['#323232', '#fbb61a', '#ff2225', '#bcaa54', '#111c6d', '#e377c2', '#4422ff']

alpha = 0.7
colors = [
    f'rgba(50,50,50,{alpha})',   # #323232
    f'rgba(251,182,26,{alpha})', # #fbb61a
    f'rgba(255,34,37,{alpha})',  # #ff2225
    f'rgba(188,170,84,{alpha})', # #bcaa54
    f'rgba(17,28,109,{alpha})',  # #111c6d
    f'rgba(227,119,194,{alpha})',# #e377c2
    f'rgba(68,34,255,{alpha})'   # #4422ff
]



link_colors = []
for (gt, pred1), row in contingency_table.iterrows():
    for pred2, value in row.items():
        link_colors.append(colors[node_mapping[gt] % 7])
        link_colors.append(colors[node_mapping[gt] % 7])
        links.append({
            'source': node_mapping[gt],  # Ground Truth to Pred1
            'target': node_mapping[pred1],
            'value': value
        })
        links.append({
            'source': node_mapping[pred1],  # Pred1 to Pred2
            'target': node_mapping[pred2],
            'value': value
        })

links_df = pd.DataFrame(links)
print(links_df)

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
# link_colors = [colors[i % 7] for i in range(len([link['source'] for link in links]))]

sankey_fig = go.Figure(go.Sankey(
    arrangement='snap',
    node=dict(
        # color=['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown']
        color = colors * 3,
        pad=80,
        thickness=100,
        line=dict(color="black", width=3),
        # label=labels,

    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links],
        color=link_colors,
    )
))
sankey_fig.update_traces(textfont=dict(
        family="Arial, sans-serif",
        size=40,
        color="black",
    ))

# Manually add annotations to place labels inside the nodes
annotations = [
    dict(
        x=0.001, y=1.002,
        text="akiec gt", showarrow=False, font=dict(size=25, color="white")
    ),
    dict(
        x=0.5, y=1.005,
        text="akiec h", showarrow=False, font=dict(size=25, color="white")
    ),
    dict(
        x=1., y=1.002,
        text="akiec nh", showarrow=False, font=dict(size=25, color="white")
    ),
    dict(
        x=0.004, y=0.915,
        text="bcc gt", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.5, y=0.92,
        text="bcc h", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.994, y=0.917,
        text="bcc nh", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.005, y=0.795,
        text="bkl gt", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.5, y=0.785,
        text="bkl h", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.993, y=0.79,
        text="bkl nh", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.005, y=0.678,
        text="df gt", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.5, y=0.644,
        text="df h", showarrow=False, font=dict(size=23, color="black")
    ),
    dict(
        x=0.99, y=0.655,
        text="df nh", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.003, y=0.565,
        text="mel gt", showarrow=False, font=dict(size=25, color="white")
    ),
    dict(
        x=0.5, y=0.53,
        text="mel h", showarrow=False, font=dict(size=23, color="white")
    ),
    dict(
        x=0.994, y=0.545,
        text="mel nh", showarrow=False, font=dict(size=25, color="white")
    ),
    dict(
        x=0.005, y=0.26,
        text="nv gt", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.5, y=0.24,
        text="nv h", showarrow=False, font=dict(size=23, color="black")
    ),
    dict(
        x=0.991, y=0.25,
        text="nv nh", showarrow=False, font=dict(size=25, color="black")
    ),
    dict(
        x=0.001, y=-0.002,
        text="vasc gt", showarrow=False, font=dict(size=25, color="white")
    ),
    dict(
        x=0.5, y=-0.002,
        text="vasc h", showarrow=False, font=dict(size=23, color="white")
    ),
    dict(
        x=0.999, y=-0.004,
        text="vasc nh", showarrow=False, font=dict(size=25, color="white")
    ),


]

# Add annotations to the figure
sankey_fig.update_layout(annotations=annotations)



# Show the figure
sankey_fig.update_layout(title_text="Sankey Diagram (Ground Truth ↦ Hair ↦ No Hair)", font_size=35, title_x=0.5)
sankey_fig.write_image('sankey_diagram.png', width=2400, height=1600)

# %%

