# %% IMPORTS AND SETTINGS
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils
import logging
from scipy.stats import chi2_contingency
from itertools import combinations
import statsmodels.stats.multitest as smm

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

to_analyze = 'test' # 'train' or 'test'

if to_analyze == 'train':
    metadata_path = config['dir']['new_metadata']
    df = pd.read_csv(metadata_path)
elif to_analyze == 'test':
    metadata_path = config['dir']['new_metadata_test']
    df = pd.read_csv(metadata_path)
    df = df[df.columns[:-2]]
    df = df[df['image_id'] != 'ISIC_0035068'] # it is not present in the test set

title_suffix = 'train set' if to_analyze == 'train' else 'test set'


cols =  ['hair', 'ruler_marks', 'bubbles', 'vignette', 'frame', 'other']
df[cols] = df[cols].fillna(0)
df[cols] = df[cols].astype(int)
# aggregate the 'other' category with the 'frame' category
df['other'] = df['frame'] | df['other']
df.drop(columns=['frame'], inplace=True)
cols =  ['hair', 'ruler_marks', 'bubbles', 'vignette', 'other']
# %% VISUALIZATION
counts = df.groupby('dx')[cols].agg(['sum', 'count'])
counts.columns = ['_'.join(col) for col in counts.columns]

fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 4)

# Axes definitions
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3]),
]

for i, ax in enumerate(axes):
    col = cols[i]
    percentage = counts[f'{col}_sum'] / counts[f'{col}_count'] * 100
    print(f"Percentage of occurrences by category for {col}:\n{round(percentage,1)}")
    order = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    sns.barplot(
        x=counts.index.str.lower(),
        y=percentage,
        order=order,
        width=0.5,
        edgecolor='black',
        facecolor=(1, 1, 1, 0),
        ax=ax
    )
    if col == 'hair':
        title = 'Hair'
    elif col == 'ruler_marks':
        title = 'Ruler Marks'
    elif col == 'bubbles':
        title = 'Interface Fluid'
    elif col == 'vignette':
        title = 'Vignette'
    if col == 'hair':
        title = 'Hair'
        ax.set_ylim(0, 90)
    elif col == 'ruler_marks':
        title = 'Ruler Marks'
        ax.set_ylim(0, 45)
    elif col == 'bubbles':
        title = 'Interface Fluid'
        ax.set_ylim(0, 35)
    elif col == 'vignette':
        title = 'Vignette'
        ax.set_ylim(0, 30)
    ax.set_title(title, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.8)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.suptitle(f'Occurrence by category in {title_suffix} [%]', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
plt.show()
fig.savefig(f'./figures/occurrence_by_category_{title_suffix}.eps', format='eps')

# %% STATISTICAL ANALYSIS
# cols =  ['hair', 'ruler_marks', 'bubbles', 'vignette', 'frame', 'other']
cols_to_test =  ['hair', 'ruler_marks', 'bubbles', 'vignette']

def chi_squared_post_hoc(df, test_column, significance_threshold=0.05):
    contingency_table = pd.crosstab(df['dx'], df[test_column])
    # make order = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    contingency_table = contingency_table.reindex(['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'])
    print(contingency_table)
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=True)
    print(dof)
    print(f"Chi-Squared Test for '{test_column}' Result:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")
    print(f"Expected Frequencies:\n{expected}")
    effect_size = np.sqrt(chi2 / (len(df) * (min(contingency_table.shape) - 1)))
    print(f"Cramer's V Effect Size: {effect_size:.4f}")
    if p < 0.05:
        combinations_of_classes = list(combinations(contingency_table.index, 2))
        p_values = []
        corrected_p_values = []

        for (i, j) in combinations_of_classes:
            pair_table = contingency_table.loc[[i, j], :]
            chi2, p, dof, expected = chi2_contingency(pair_table, correction=True)
            p_values.append(p)
        
        _, corrected_p_values, _, _ = smm.multipletests(p_values, method='bonferroni')

        # Create a DataFrame for visualization
        post_hoc_results = pd.DataFrame({
            'Class 1': [i[0] for i in combinations_of_classes],
            'Class 2': [i[1] for i in combinations_of_classes],
            'Corrected P-value': corrected_p_values
        })
        
        print("\nPost-Hoc Pairwise Chi-Squared Tests (Bonferroni Corrected):")
        for (i, j), corrected_p in zip(combinations_of_classes, corrected_p_values):
            print(f"Comparison between {i} and {j}: corrected p-value = {corrected_p:.4f}")

        corrected_p_value_matrix = np.full((len(contingency_table.index), len(contingency_table.index)), np.nan)
        
        for idx, (i, j) in enumerate(combinations_of_classes):
            class_i_idx = contingency_table.index.get_loc(i)
            class_j_idx = contingency_table.index.get_loc(j)
            
            # assign value if p-value is below the signif th
            # if corrected_p_values[idx] < significance_threshold:
            corrected_p_value_matrix[class_i_idx, class_j_idx] = corrected_p_values[idx]
            corrected_p_value_matrix[class_j_idx, class_i_idx] = corrected_p_values[idx]  # symmetric matrix

        # masking upper triangle (above diagonal) and diagonal of the matrix
        mask = np.triu(np.ones_like(corrected_p_value_matrix, dtype=bool))

        # redundant row and col deleted
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(corrected_p_value_matrix, annot=True, fmt='.2f', cbar=False,
                    cmap='gray', square=True,
                    annot_kws={'size': 16},
                    cbar_kws={'label': 'Corrected P-value'},
                    linewidths=.5,
                    linecolor='black',
                    xticklabels=contingency_table.index, yticklabels=contingency_table.index,
                    # mask=mask[1:, :-1],
                    )
        # sns.heatmap(corrected_p_value_matrix[1:, :-1], annot=True, fmt='.4f', cbar=False, 
        #             cmap='gray', square=True,
        #             xticklabels=contingency_table.index[:-1], yticklabels=contingency_table.index[1:],
        #             # mask=mask[1:, :-1], 
        #             cbar_kws={'label': 'Corrected P-value'})
        if col == 'hair':
            title = 'Hair'
        elif col == 'ruler_marks':
            title = 'Ruler Marks'
        elif col == 'bubbles':
            title = 'Interface Fluid'
        elif col == 'vignette':
            title = 'Vignette'
        plt.title(f"{title} in " + title_suffix, fontsize=16)
        # plt.xlabel("Class", fontsize=14)
        # plt.ylabel("Class", fontsize=14)
        plt.xticks(rotation=0, fontsize=16)
        plt.yticks(rotation=0, fontsize=16)
        plt.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()
        plt.show()
        fig.savefig(f'./figures/{test_column}_post_hoc_{title_suffix}.png', pad_inches=0, dpi=600)

    else:
        print(f"\nNo significant difference found in the initial Chi-Squared test for '{test_column}'.")
# %%
for col in cols_to_test:
    chi_squared_post_hoc(df, col)
# %%