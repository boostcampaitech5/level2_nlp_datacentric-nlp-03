import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import os
import numpy as np

def draw_eda(val_df, test_df, OUTPUT_DIR):
    result_df = val_df.copy()
    submission_df = test_df.copy()

    category_dict = {
    0: 'IT/Science',
    1: 'Economy',
    2: 'Society',
    3: 'Life/Culture',
    4: 'Global',
    5: 'Sports',
    6: 'Politics'
    }

    fig, ax = plt.subplots(2,2, figsize=(20,20))
    result_df['correct'] = (result_df['target']==result_df['preds']).astype('uint16')

    total, correct=[],[]
    num_labels = result_df['target'].nunique()
    for i in range(num_labels):
        tmp = result_df[result_df['target']==i]
        total.append(tmp['correct'].count())
        correct.append(tmp['correct'].sum())

    ax[0,0].bar(range(num_labels), total, color='navy', label='total', width=0.6)
    ax[0,0].bar(range(num_labels), correct, color='orange',label='correct', width=0.6)
    for i in range(num_labels):
        ax[0,0].text(i,total[i],f"{correct[i]/total[i]:.3f}", ha='center', va='bottom')
    ax[0,0].set_xticks(list(category_dict.keys()), list(category_dict.values()), rotation=-15)
    ax[0,0].set_title('Correct ratio')
    ax[0,0].legend()

    cross_df = pd.crosstab(result_df['target'].map(category_dict), result_df['preds'].map(category_dict)).reindex(category_dict.values())
    cross_df = cross_df.reindex(columns=category_dict.values())
    # normalized_df = cross_df.apply(lambda x: 100*(x-x.min())/(x.max()-x.min()), axis=0)
    normalized_df = cross_df.apply(lambda x: 100*x / x.sum(), axis=1)

    sns.heatmap(cross_df, annot=True, fmt='g', cmap='OrRd', linewidths=0.8, ax=ax[1,0])
    ax[1,0].set_title('Heatmap')
    ax[1,0].set_xticklabels(list(category_dict.values()), rotation=15)
    ax[1,0].set_yticklabels(list(category_dict.values()), rotation=15)

    sns.heatmap(normalized_df, annot=True, fmt='.3g', cmap='OrRd', linewidths=0.8, ax=ax[1,1])
    ax[1,1].set_title('Normalized Heatmap')
    ax[1,1].set_xticklabels(list(category_dict.values()), rotation=15)
    ax[1,1].set_yticklabels(list(category_dict.values()), rotation=15)

    pred_count = submission_df['target'].value_counts().sort_index()
    ax[0,1].bar(pred_count.index, pred_count, width=0.6, color='skyblue', edgecolor='black')
    for i, v in category_dict.items():
        ax[0,1].text(i, pred_count[i], str(pred_count[i]), ha='center', va='bottom')
    ax[0,1].set_xticks(list(category_dict.keys()), list(category_dict.values()), rotation=15)
    ax[0,1].set_title('submission')

    fig.savefig(os.path.join(OUTPUT_DIR, 'figure.png'))
    fig.clf()
    plt.close()
    print('figure created')

    return np.round(np.array(correct) / np.array(total),3).tolist()