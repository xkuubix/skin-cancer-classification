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
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patheffects import withStroke

def tabular_explanations(finfo, attributions, ax):
    feature_names = finfo['selected_features']
    feature_names = [name.replace('original_', '') for name in feature_names]
    attr = attributions[1].squeeze().cpu().numpy()
    importance_scores = np.abs(attr)
    topk = 10
    top_indices = np.argsort(importance_scores)[-topk:][::-1]
    top_scores = importance_scores[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    plot_features = top_features
    plot_scores = list(top_scores)
    colors = []
    for feature in plot_features:
        if feature.endswith('_b'):
            colors.append('blue')
        elif feature.endswith('_g'):
            colors.append('green')
        elif feature.endswith('_r'):
            colors.append('red')
        else:
            colors.append('gray')
    sns.barplot(x=plot_scores, y=plot_features, palette=colors, ax=ax,
                hue=plot_features,
                legend=True,
                dodge=False,
                gap=0
                )
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in ['blue', 'green', 'red', 'gray']]
    labels = ['Blue', 'Green', 'Red', 'Grayscale']
    ax.legend(handles, labels, title='Channel', loc='best', fontsize=12, title_fontsize=16)
    for i, feature in enumerate(plot_features):
        # ax.text(plot_scores[i], i, f' {feature}', va='center') # start outside right
        ax.text(0, i, f' {feature}', va='center',
                color='white',  # Font color
                fontsize=16,  # Font size
                path_effects=[
                withStroke(linewidth=3, foreground='black')  # Outline effect
                ]) # start inside bars
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

def plot_attr(data, attributions, pred, prob):
    path = data['img_path']
    original_image = np.array(Image.open(path[0]))
    path = data['seg_path']
    segmentation_image = np.array(Image.open(path[0]))
    processed_image = image[0].cpu().numpy().transpose(1, 2, 0)
    attr = attributions[0][0].cpu().numpy().transpose(1, 2, 0)
    id = data['img_path'][0].split('/')[-1].split('.')[0]
    default_cmap = LinearSegmentedColormap.from_list('custom red',
                                                    [(0, '#000000'),
                                                     (.5, '#ff0000'),
                                                    (1, '#ffff00')], N=256)
    seg_cmap = LinearSegmentedColormap.from_list('custom bin',
                                                 [(0, '#000000'),
                                                 (1, '#ffffff')], N=2)

    # fig, ax = plt.subplots(1, 5, figsize=(18, 9))
    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    ax = [plt.subplot(2,3,1), plt.subplot(2,3,2), plt.subplot(2,3,4), plt.subplot(2,3,5),
          plt.subplot(1,3,3)]

    _ = viz.visualize_image_attr(None, original_image,
                                 method="original_image",
                                 title=f'Original Image ID:{id}',
                                 use_pyplot=False,
                                 plt_fig_axis=(fig, ax[0]))
    _ = viz.visualize_image_attr(None, processed_image,
                                 method="original_image",
                                 title='Processed Image',
                                 use_pyplot=False,
                                 plt_fig_axis=(fig, ax[1]))
    ax[2].imshow(segmentation_image, cmap=seg_cmap)
    ax[2].set_title('Segmentation Mask')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    for spine in ax[2].spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    ax[2].spines['top'].set_visible(True)
    ax[2].spines['bottom'].set_visible(True)
    ax[2].spines['left'].set_visible(True)
    ax[2].spines['right'].set_visible(True)

    _ = viz.visualize_image_attr(attr,
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 title=f'Attributions (pred:{pred.lower()}, p:{prob})',
                                 use_pyplot=False,
                                 plt_fig_axis=(fig, ax[3]))
    
    tabular_explanations(finfo, attributions, ax[4])
    ax[4].set_title(f'Top {10} Important Radiomics')
    ax[4].set_xlabel('Importance Score')
    for spine in ax[2].spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['bottom'].set_visible(False)
    ax[4].spines['left'].set_visible(False)
    ax[4].spines['right'].set_visible(False)
    ax[4].tick_params(left=False, labelleft=False)
    ax[4].set_axisbelow(True)
    ax[4].grid(axis='x', linestyle='-', linewidth=1, zorder=0)

    for a in ax:
        a.title.set_fontsize(16)

    fig.tight_layout()
    plt.close()
    return fig

# dev
# fig = plot_attr(data, attributions, pred, prob)

# %%
# with hair
paths = ['model_fold_0_f667fabfa1c747bc85e49b1e0b7dfe3e',
         'model_fold_1_dbca0643b5054fe9a7d77ac2f0dcb86e',
         'model_fold_2_6fdc181b2f5d4772a3a9b5f43b75b7d5',
         'model_fold_3_73591c86886046a3af9877610bcf56ff',
         'model_fold_4_dc73e29eb99f4297bfb20d116f3a7e6a'
         ]
# no hair
paths = ['model_fold_0_e337ad33562b4c61aaa956ef1f07befa',
         'model_fold_1_9dff589f89c743448e1405d58daf5c38',
         'model_fold_2_f58823fc62934ce7b8659ea4c85cbb59',
         'model_fold_3_45d4591cd68d453e8ddbbdd4b050dd3a',
         'model_fold_4_743f42ba09ce44ef8b8db3cfc533a485'
         ]

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
# %% ---------------------K-Fold Cross Validation---------------------
fold_results = {}
k = config['dataset']['k_folds']
# kf = KFold(n_splits=k, shuffle=True, random_state=seed)
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
# Concatenate train and validation set for k-fold cross validation
df = pd.concat([train_df, val_df], ignore_index=True)
kf.get_n_splits(df)
for fold_index, (train_indices, val_indices) in enumerate(kf.split(df, df['dx'])):

    train_fold = df.iloc[train_indices]
    val_fold = df.iloc[val_indices]
    train_fold, val_fold, test_fold, finfo = utils.prepare_data_for_fold(
        train_fold.copy(), val_fold.copy(), test_df.copy(), random_state=seed)
    test_ds = HAM10000(df=test_fold, transform=transforms_val_test, mode=config['dataset']['mode'])
    
    mapping_handler = MappingHandler().mapping
    labels = [int(mapping_handler[label.upper()]) for label in train_fold['dx'].values]

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1,
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
    model.eval()
    explainer = IntegratedGradients(model)
    with torch.no_grad():
        i = 0
        i_max = 5
        for data in test_dl:
            if config['dataset']['mode'] in ['images', 'hybrid']:  
                image = data['image'].to(device)
            if config['dataset']['mode'] in ['radiomics', 'hybrid']:
                radiomic_features = data['features'].to(device)
            else:
                radiomic_features = None
            target_idx = data['label'].to(device)
            image_baseline = torch.rand(image.size()).to(device)
            # image_baseline = torch.zeros_like(image).to(device)
            tabular_baseline = torch.zeros_like(radiomic_features).to(device)
            attributions = explainer.attribute((image, radiomic_features),
                                               baselines=(image_baseline, tabular_baseline),
                                               target=target_idx,
                                               n_steps=200
                                               )
            output = explainer.forward_func(image, radiomic_features)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            prob = prob.reshape(-1)[pred].item().__round__(2)
            pred = next((k for k, v in mapping_handler.items() if v == pred), None)
            fig = plot_attr(data, attributions, pred, prob)
            id = data['img_path'][0].split('/')[-1].split('.')[0]
            path = config['dir']['results'] + f'no_hair/{id}_attr.png'
            # path = config['dir']['results'] + f'hair/{id}_attr.png'
            fig.savefig(path)
            i += 1
            print(f'{i}/1511')
            # if i >= i_max:
            #     break
            # break
    break # only one fold for now

# %% 
