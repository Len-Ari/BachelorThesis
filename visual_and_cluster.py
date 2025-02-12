# All visualization functions used in the python notebook are saved here

import matplotlib.pyplot as plt
import numpy as np
import os
import json

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import FeatureAgglomeration


def plot_reduce_dimensions(embeddings, methods=['pca'], labels=None, has_unlabeled=False, query_embeddings=None , cmap='tab20', model_name=None):
    '''
    Reduce the dimensionality of embeddings using the method defined by methods.
    If labels are provided use the cmap to colorcode the resulting scatterplot.

    Parameters:
    - embeddings (ndarray): Array of embeddings to reduce to two dimensions.
    - methods (list<String>, optional): A List of methods to be used for the dimensionality reduction.
        ('pca', 'pca_tsne', 'agglo_tsne')
    - labels (ndarray, optional): Labels for colorcoding.
    - has_unlabeled (bool, optional): If True labels with value 0 mapped to 'No Label' and rest to label-1.
    - query_embeddings (ndarray, optional): If not None draw markers with label, where query embeddings are mapped.
    - cmap (matplotlib.cm, optional): A matplotlib colomap to be used for label colors.
    '''
    ret = []
    search_labels = labels if labels is not None else np.ones(embeddings.squeeze().shape[0])
    emb_len = np.squeeze(embeddings).shape[0]
    n = len(methods)
    fig, axs= plt.subplots(n)
    fig.set_size_inches(8, 8*n)
    labels = list(map(str, range(0, len(set(search_labels)))))
    if has_unlabeled:
        labels = ['No Label'] + labels[:-1]
    if query_embeddings is not None:
        embeddings = np.append(np.squeeze(embeddings), query_embeddings, axis = 0)
        search_labels.extend(['0'] * len(query_embeddings))

    if 'pca' in methods:
        ax = axs[methods.index('pca')]  if n > 1 else axs

        pca_2 = PCA(n_components=2, random_state=42)
        pca_2_results= pca_2.fit_transform(np.squeeze(embeddings))
        ax.set_title('PCA-Reduction to 2 dimensions')
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax0 = ax.scatter(pca_2_results[:, 0], pca_2_results[:, 1], marker=".", c=search_labels, cmap=cmap)

        if len(labels)>1:
            handles, _ = ax0.legend_elements()
            legend = ax.legend(handles=handles, labels=labels, title="Label Index")
            ax.add_artist(legend)
        ret.append(pca_2_results[:emb_len])

        if query_embeddings is not None:
            query_2d = pca_2_results[-len(query_embeddings):]
            ax.scatter(query_2d[:, 0], query_2d[:, 1], marker="o", c="white",alpha=0.2, s=200, edgecolor = "k")
            for i, q in enumerate(query_embeddings):
                ax.scatter(query_2d[i][0], query_2d[i][1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    if 'pca_tsne' in methods:
        ax = axs[methods.index('pca_tsne')]  if n > 1 else axs

        pca_50 = PCA(n_components=50, random_state=42)
        pca_50_result = pca_50.fit_transform(np.squeeze(embeddings))

        tsne = TSNE(random_state=42)
        tsne_pca_results = tsne.fit_transform(pca_50_result)
        ax.set_title('PCA-Reduction to 50 dimensions\n+T-SNE reduction to 2 dimensions')
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax1 = ax.scatter(tsne_pca_results[:, 0], tsne_pca_results[:, 1], marker=".", c=search_labels, cmap=cmap)
        
        if len(labels)>1:
            handles, _ = ax1.legend_elements()
            legend1 = ax.legend(handles=handles, labels=labels, title="Label Index")
            ax.add_artist(legend1)
        ret.append(tsne_pca_results[:emb_len])

        if query_embeddings is not None:
            query_2d = tsne_pca_results[-len(query_embeddings):]
            ax.scatter(query_2d[:, 0], query_2d[:, 1], marker="o", c="white",alpha=0.2, s=200, edgecolor="k")
            for i, q in enumerate(query_embeddings):
                ax.scatter(query_2d[i][0], query_2d[i][1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")


    if 'agglo_tsne' in methods:
        ax = axs[methods.index('agglo_tsne')]  if n > 1 else axs

        # Feature Agglomeration combines similar features
        # Should only be used to reduce dimensionality to decent amount 50-100
        agglo_50 = FeatureAgglomeration(n_clusters=50)
        agglo_50_results = agglo_50.fit_transform(np.squeeze(embeddings))
        #Combine with T-SNE to reduce to 2 dimensions
        tsne = TSNE(random_state=42)
        tsne_agglo_50_results = tsne.fit_transform(agglo_50_results)
        ax.set_title('Feature agglomeration to 50 dimensions\n+T-SNE reduction to 2 dimensions')
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax2 = ax.scatter(tsne_agglo_50_results[:, 0], tsne_agglo_50_results[:, 1], marker=".", c=search_labels, cmap=cmap)

        if len(labels)>1:
            handles, _ = ax2.legend_elements()
            legend2 = ax.legend(handles=handles, labels=labels, title="Label Index")
            ax.add_artist(legend2)
        ret.append(tsne_agglo_50_results[:emb_len])

        if query_embeddings is not None:
            query_2d = tsne_agglo_50_results[-len(query_embeddings):]
            ax.scatter(query_2d[:, 0], query_2d[:, 1], marker="o", c="white",alpha=0.2, s=200, edgecolor="k")
            for i, q in enumerate(query_embeddings):
                ax.scatter(query_2d[i][0], query_2d[i][1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    if model_name is not None:
        plt.savefig(f'Plots/DimReduction/{methods[0]}_{model_name}.png')
    plt.show()
    return ret


def label_from_metadata(id_list:list, dir_path:list, field_name:str, labels=None)->dict:
    "Find metadata for the given Id's and return list for specified field. This only works with single value fields with path metadata[field_name]!"
    if labels == None:
        labels = dict.fromkeys(id_list)

    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file():
                # Check if file is json
                _, extension = os.path.splitext(entry.path)
                if extension == '.jsonl':
                    # Read in Jsonl file
                    with open(entry.path, 'r', encoding='utf-8') as file_in:
                        for line in file_in:
                            article_data = json.loads(line)
                            if (article_data['metadata']['article_ids']['pmc'] in id_list):
                                # Extract values for column here
                                labels[article_data['metadata']['article_ids']['pmc']] = article_data['metadata'][field_name]
                                continue
                            if (article_data['filename'] in id_list):
                                # Need this in case I used filename as reference...
                                labels[article_data['filename']] = article_data['metadata'][field_name]
            elif entry.is_dir():
                # Nested Directories
                labels = label_from_metadata(id_list, entry.path, field_name, labels)

    #assert len(labels) == len(id_list)
    return labels