import json
import numpy as np
import nltk.data
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
import random


def url_to_corpus(url,mode):
  
    """
    Given a JSON file containing a list of dictionaries with at least one key being mode,
    preprocesses and tokenizes the values associated with mode.

    Parameters:
    url (str): URL of the JSON file
    mode (str): key of the dictionary in the JSON file to preprocess and tokenize

    Returns:
    docs (np.array): array of preprocessed and tokenized documents
    types (set): set of unique tokens found in the documents
    """
    
    nltk.download('stopwords')
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    stop_words = stopwords.words('english')

    with open(url, 'r') as f: data = json.load(f)

    num_docs = len(data)
    docs = np.empty(num_docs, dtype='object')

    for i in range(num_docs): docs[i] = data[i][mode]

    # Preprocessing
    docs = [[[word for word in list(TextBlob(doc).words) if word not in stop_words] for doc in sent_detector.tokenize(d.strip())] for d in docs]

    # Appending sentences to a single token document
    merged_doc = []
    types = set()

    # Merge all sentences in one single token document
    for doc in docs:
        temp_doc = []
        for sentence in doc:
            temp_doc+=sentence
            for word in sentence:
                types.add(word)
        merged_doc.append(temp_doc)

    docs = merged_doc
    return docs,types

def docs_to_embeddings(docs, model):

    """
    Given an array of preprocessed and tokenized documents and a word embedding model,
    returns the embeddings of each document.

    Parameters:
    docs (np.array): array of preprocessed and tokenized documents
    model (gensim.models.keyedvectors.Word2VecKeyedVectors): word embedding model

    Returns:
    document_embeddings (np.array): array of document embeddings
    """

    docs = [' '.join(doc) for doc in docs]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs)

    vocabulary = vectorizer.vocabulary_
    idf_values = vectorizer.idf_

    max_length = max(len(doc) for doc in docs)
    embeddings = np.zeros((len(docs), max_length, model.vector_size))

    for i, doc in enumerate(docs):
        for j, word in enumerate(doc):
            embedding = model[word] if word in model else np.zeros(model.vector_size)
            embeddings[i, j, :] = embedding

    weights = np.zeros((len(docs), max_length))
    for i, doc in enumerate(docs):
        tokens = doc.split()
        for j, token in enumerate(tokens):
            if token in vocabulary:
                token_index = vocabulary[token]
                tfidf_weight = idf_values[token_index]
                weights[i, j] = tfidf_weight

    
    document_embeddings = np.zeros((len(docs), model.vector_size))
    for i, doc in enumerate(docs):
        for j, word in enumerate(doc):
            document_embeddings[i, :] += weights[i, j] * embeddings[i, j, :]
    
    return document_embeddings

def train(doc_embeddings , query_embeddings):

    """
    Train a neural network model to predict query relevance based on document and query embeddings.

    Args:
    doc_embeddings (np.array): Array of document embeddings.
    query_embeddings (np.array): Array of query embeddings.

    Returns:
    None

    Plots:
    Two subplots showing the training and validation loss and accuracy over epochs.

    """

    with open("cranfield\cran_qrels.json", "r") as f:
        data = json.load(f)

    y = tf.one_hot(np.array([data[i]['position'] for i in range(len(data))]),depth=4)
    x_doc_seq = np.array([doc_embeddings[int(data[i]['id'])-1] for i in range(len(data))])
    x_query_seq = np.array([query_embeddings[int(data[i]['query_num'])-1] for i in range(len(data))])

    doc_shape = doc_embeddings.shape[1:]
    query_shape = query_embeddings.shape[1:]

    doc_inputs = tf.keras.layers.Input(shape=doc_shape)
    query_inputs = tf.keras.layers.Input(shape=query_shape)

    doc = tf.keras.layers.Dense(128, activation='relu')(doc_inputs)
    doc = tf.keras.layers.Dense(128, activation='relu')(doc)
    query = tf.keras.layers.Dense(128, activation='relu')(query_inputs)
    query = tf.keras.layers.Dense(128, activation='relu')(query)

    x = tf.keras.layers.Concatenate()([doc, query])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    opt = Adam(learning_rate=1e-6)

    model = tf.keras.Model(inputs=[doc_inputs, query_inputs], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit([x_doc_seq, x_query_seq], y, epochs=20, batch_size=4, validation_split=0.2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot training and validation accuracy
    axs[1].plot(history.history['accuracy'], label='Training Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

def clustering(X, random_state, r):

    # Spectral clustering takes too muc time.
    """
    This function performs clustering on the input data X using Gaussian Mixture Model and evaluates the performance of clustering using Silhouette score, Davies-Bouldin index, and Calinski-Harabasz index. It also plots these evaluation metrics and the minimum and maximum number of documents per cluster for different number of clusters.

    Parameters:

        X: input data to be clustered
        random_state: random state for Gaussian Mixture Model
        r: range of number of clusters to evaluate performance

    Returns:
    None

    Plots:

        Plot 1: Number of clusters vs Silhouette score
        Plot 2: Number of clusters vs Davies-Bouldin index
        Plot 3: Number of clusters vs Calinski-Harabasz index
        Plot 4: Number of clusters vs Average score (average of Silhouette score, Calinski-Harabasz index, and (1 - Davies-Bouldin index))
        Plot 5: Number of clusters vs Minimum number of documents per cluster
        Plot 6: Number of clusters vs Maximum number of documents per cluster
    """

    n_clusters_range = r
    silhouette_scores_gmm = []
    silhouette_scores_dbscan = []
    db_scores_gmm = []
    db_scores_dbscan = []
    ch_scores_gmm = []
    ch_scores_dbscan = []
    min_docs_gmm = []
    min_docs_dbscan = []
    max_docs_gmm = []
    max_docs_dbscan = []
    K = []

    for n_clusters in tqdm(n_clusters_range):
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, max_iter=500)
        sc = SpectralClustering(n_clusters=n_clusters, random_state=random_state)

        gmm.fit(X)
        sc.fit(X)

        cluster_labels_gmm = gmm.predict(X)
        cluster_labels_dbscan = sc.labels_

        silhouette_scores_gmm.append(silhouette_score(X, cluster_labels_gmm))
        silhouette_scores_dbscan.append(silhouette_score(X, cluster_labels_dbscan))

        db_scores_gmm.append(davies_bouldin_score(X, cluster_labels_gmm))
        db_scores_dbscan.append(davies_bouldin_score(X, cluster_labels_dbscan))

        ch_scores_gmm.append(calinski_harabasz_score(X, cluster_labels_gmm))
        ch_scores_dbscan.append(calinski_harabasz_score(X, cluster_labels_dbscan))

        doc_indices_gmm = {i: np.where(cluster_labels_gmm == i)[0].tolist() for i in set(cluster_labels_gmm)}
        min_docs_per_cluster = min(len(docs) for docs in doc_indices_gmm.values())
        max_docs_per_cluster = max(len(docs) for docs in doc_indices_gmm.values())
        min_docs_gmm.append(min_docs_per_cluster)
        max_docs_gmm.append(max_docs_per_cluster)

        doc_indices_dbscan = {i: np.where(cluster_labels_dbscan == i)[0].tolist() for i in set(cluster_labels_dbscan)}
        min_docs_per_cluster = min(len(docs) for docs in doc_indices_dbscan.values())
        max_docs_per_cluster = max(len(docs) for docs in doc_indices_dbscan.values())
        min_docs_dbscan.append(min_docs_per_cluster)
        max_docs_dbscan.append(max_docs_per_cluster)


        K.append(n_clusters)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

    silhouette_scores_norm_gmm = (silhouette_scores_gmm - np.min(silhouette_scores_gmm)) / (np.max(silhouette_scores_gmm) - np.min(silhouette_scores_gmm))
    db_scores_norm_gmm = (db_scores_gmm - np.min(db_scores_gmm)) / (np.max(db_scores_gmm) - np.min(db_scores_gmm))
    ch_scores_norm_gmm = (ch_scores_gmm - np.min(ch_scores_gmm)) / (np.max(ch_scores_gmm) - np.min(ch_scores_gmm))
    avg_scores_gmm = (silhouette_scores_norm_gmm + ch_scores_norm_gmm + (1 - db_scores_norm_gmm)) / 3

    silhouette_scores_norm_dbscan = (silhouette_scores_dbscan - np.min(silhouette_scores_dbscan)) / (np.max(silhouette_scores_dbscan) - np.min(silhouette_scores_dbscan))
    db_scores_norm_dbscan = (db_scores_dbscan - np.min(db_scores_dbscan)) / (np.max(db_scores_dbscan) - np.min(db_scores_dbscan))
    ch_scores_norm_dbscan = (ch_scores_dbscan - np.min(ch_scores_dbscan)) / (np.max(ch_scores_dbscan) - np.min(ch_scores_dbscan))
    avg_scores_dbscan = (silhouette_scores_norm_dbscan + ch_scores_norm_dbscan + (1 - db_scores_norm_dbscan)) / 3

    ax[0, 0].plot(n_clusters_range, silhouette_scores_gmm, 'bo-', label = "GMM")
    ax[0, 0].plot(n_clusters_range, silhouette_scores_dbscan, 'ro-', label = "SC")
    ax[0, 0].set_xlabel('Number of clusters')
    ax[0, 0].set_ylabel('Silhouette score')
    ax[0, 1].plot(n_clusters_range, db_scores_gmm, 'bo-', label = "GMM")
    ax[0, 1].plot(n_clusters_range, db_scores_dbscan, 'ro-', label = "SC")
    ax[0, 1].set_xlabel('Number of clusters')
    ax[0, 1].set_ylabel('Davies-Bouldin index')
    ax[1, 0].plot(n_clusters_range, ch_scores_gmm, 'bo-', label = "GMM")
    ax[1, 0].plot(n_clusters_range, ch_scores_dbscan, 'ro-', label = "SC")
    ax[1, 0].set_xlabel('Number of clusters')
    ax[1, 0].set_ylabel('Calinski-Harabasz index')
    ax[1, 1].plot(n_clusters_range, avg_scores_gmm, 'bo-', label = "GMM")
    ax[1, 1].plot(n_clusters_range, avg_scores_dbscan, 'ro-', label = "SC")
    ax[1, 1].set_xlabel('Number of clusters')
    ax[1, 1].set_ylabel('Average Score')
    ax[2, 0].plot(n_clusters_range, min_docs_gmm, 'bo-', label = "GMM")
    ax[2, 0].plot(n_clusters_range, min_docs_dbscan, 'ro-', label = "SC")
    ax[2, 0].set_xlabel('Number of clusters')
    ax[2, 0].set_ylabel('Minimum docs per cluster')
    ax[2, 1].plot(n_clusters_range, max_docs_gmm, 'bo-', label = "GMM")
    ax[2, 1].plot(n_clusters_range, max_docs_dbscan, 'ro-', label = "SC")
    ax[2, 1].set_xlabel('Number of clusters')
    ax[2, 1].set_ylabel('Maximum docs per cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()



def pickel_dictionary(data , url , k, random_state):

    gmm = GaussianMixture(n_components=k, max_iter = 500, random_state=random_state)
    gmm.fit(data)

    doc_clusters = gmm.predict(data)
    centroids = gmm.means_
    doc_indices = {tuple(centroid): np.where(doc_clusters == i)[0].tolist() for i, centroid in enumerate(centroids)}

    centroid_mapping_svd = doc_indices

    min_docs_per_cluster = 9999

    for i in range(k):

        min_docs_per_cluster = min(min_docs_per_cluster,(len(centroid_mapping_svd[list(centroid_mapping_svd.keys())[i]])))

    with open(url, 'wb') as f:
        pickle.dump(centroid_mapping_svd, f)

    

def cluster_2d(data, k, random_state):

    """
    This function performs clustering on 2D data using Gaussian Mixture Model and visualizes the results using PCA and t-SNE.

    Args:
    data: 2D data to be clustered
    k: Number of clusters to be formed
    random_state: Seed value for reproducibility of results

    Returns:
    None (Displays the clustering results using PCA and t-SNE plots)

    Plots:
    - Scatter plot of the clustering results using PCA
    - Scatter plot of the clustering results using t-SNE

    """

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    gmm = GaussianMixture(n_components=k, max_iter=500, random_state=random_state)
    gmm.fit(data)

    labels = gmm.predict(data )

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data )

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=random_state)
    X_tsne = tsne.fit_transform(data )
    colors = np.random.rand(k, 3)
    cmap = ListedColormap(colors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.5)
    ax1.set_title("Clustering Results using EM and PCA")

    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=cmap, alpha=0.5)
    ax2.set_title("Clustering Results using EM and t-SNE")

    plt.show()

def calculate_metrics(q_retrieved, q_rel, k_values):
    """
    Function: calculate_metrics

    This function calculates the metrics MAP (Mean Average Precision) and nDCG (Normalized Discounted Cumulative Gain)
    for different k values based on retrieved and relevant documents for a given set of queries.

    Inputs:
        q_retrieved: a dictionary where the keys are query ids and the values are lists of retrieved documents
        q_rel: a dictionary where the keys are query ids and the values are sets of relevant documents
        k_values: a list of integers representing the different k values to calculate the metrics for

    Returns:
        map_values: a numpy array containing the MAP values for the different k values
        ndcg_values: a numpy array containing the nDCG values for the different k values
        precision: a list of precision values for the different k values
        recall: a list of recall values for the different k values
    """

    # Initialize arrays to store metrics for different k values
    map_values = np.zeros(len(k_values))
    ndcg_values = np.zeros(len(k_values))
    precision = np.zeros(len(k_values))
    recall = np.zeros(len(k_values))
    
    # Loop over all queries
    for q_id in q_retrieved.keys():
        retrieved_docs = q_retrieved[q_id][:k_values[-1]] # Consider only top k retrieved documents
        
        # Calculate metrics for each value of k
        for i, k in enumerate(k_values):
            retrieved_k = retrieved_docs[:k]
            relevant_docs = q_rel[q_id]
            
            # Calculate precision, recall, and average precision
            p = sum([1 if doc in relevant_docs else 0 for doc in retrieved_k]) / k
            r = sum([1 if doc in relevant_docs else 0 for doc in retrieved_k]) / len(relevant_docs)

            ap = 0
            for j, doc in enumerate(retrieved_k):
                if doc in relevant_docs:
                    ap += sum([1 if d in relevant_docs else 0 for d in retrieved_k[:j+1]]) / (j+1)
            if len(relevant_docs) == 0:
                ap = 0
            else:
                ap /= len(relevant_docs)
            
            # Calculate nDCG
            dcg = sum([(2 if doc in relevant_docs else 1) / np.log2(j+2) for j, doc in enumerate(retrieved_k)])
            idcg = sum([(2 if doc in relevant_docs else 1) / np.log2(j+2) for j, doc in enumerate(relevant_docs)])
            if idcg == 0:
                ndcg = 0
            else:
                ndcg = dcg / idcg
            
            # Store metrics for current k value
            map_values[i] += ap
            ndcg_values[i] += ndcg
            precision[i] += p
            recall[i] += r


    # Divide by number of queries to get average metrics
    map_values /= len(q_retrieved)
    ndcg_values /= len(q_retrieved)
    precision /= len(q_retrieved)
    recall /= len(q_retrieved)
    
    # Return average metrics for different k values
    return {"map": map_values, "ndcg" : ndcg_values, "precision" : precision, "recall" : recall}

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

def gen_result_centroids(query_embeddings, doc_embeddings, centroid_mapping):

    results_dict = {}

    for i, query_embedding in enumerate(query_embeddings):
        distances = [cos_sim(query_embedding,centroid) for centroid in centroid_mapping if centroid is not "s"]
        nearest_centroid = np.argmin(distances)
        nearest_docs = centroid_mapping[list(centroid_mapping.keys())[nearest_centroid]]
        similarities = [cos_sim(doc_embedding,query_embedding) for doc_embedding in doc_embeddings[nearest_docs]]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_docs = [nearest_docs[i]+1 for i in sorted_indices]

        while(len(sorted_docs)<1200) : sorted_docs.append(0)

        results_dict[str(i+1)] = sorted_docs

    return results_dict

def gen_result_index(query_embeddings, doc_embeddings, queries, inverted_index):

    results_dict = {}

    for i, query_embedding in enumerate(query_embeddings):
        query_results = {}
        for term in queries[i]:
            if term in inverted_index:
                docs_containing_term = inverted_index[term]
                for doc_id in docs_containing_term:
                    doc_embedding = doc_embeddings[doc_id-1]
                    similarity = cos_sim(query_embedding, doc_embedding)
                    if doc_id in query_results:
                        query_results[doc_id] += similarity
                    else:
                        query_results[doc_id] = similarity
        sorted_docs = sorted(query_results, key=query_results.get, reverse=True)
        sorted_docs = [doc_id+1 for doc_id in sorted_docs]

        while(len(sorted_docs)<1200) : sorted_docs.append(0)

        results_dict[str(i+1)] = sorted_docs

    return results_dict

def plot_centroid_indexing(q_rel, Q):

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    query_url = "cranfield_embeddings\query_embeddings_svd.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_svd.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_svd.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "SVD")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "SVD")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_embeddings\query_embeddings_word2vec.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_word2vec.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_word2vec.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "W2V")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "W2V")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_sequences\q_seq_embedding.csv"
    docs_url = "cranfield_sequences\doc_seq_embedding.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_seq.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "S_1")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "S_1")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_sequences\q_seq_embedding_2.csv"
    docs_url = "cranfield_sequences\doc_seq_embedding_2.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_seq_2.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "S_2")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "S_2")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
        
    plt.tight_layout()
    plt.legend()

    query_url = "cranfield_embeddings\query_embeddings_vs.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_vs.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "Vector Space", linestyle = '--')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "Vector Space", linestyle = '--')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
        
    plt.tight_layout()
    plt.legend()

def plot_inverted_indexing(q_rel, Q):

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    query_url = "cranfield_embeddings\query_embeddings_svd.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_svd.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "SVD")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "SVD")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_embeddings\query_embeddings_word2vec.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_word2vec.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "W2V")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "W2V")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_sequences\q_seq_embedding.csv"
    docs_url = "cranfield_sequences\doc_seq_embedding.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "S_1")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "S_1")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_sequences\q_seq_embedding_2.csv"
    docs_url = "cranfield_sequences\doc_seq_embedding_2.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "S_2")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "S_2")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
        
    query_url = "cranfield_embeddings\query_embeddings_vs.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_vs.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    s = results['ndcg'][-1]

    axs[0].plot(range(2, 10), results['ndcg']*(s/results['ndcg'][-1]), label = "Vector Space", linestyle = '--')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map']*(s/results['map'][-1]), label = "Vector Space", linestyle = '--')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
        
    plt.tight_layout()
    plt.legend()

def plot_metrics(metrics, k_values):

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    
    # Plot NDCG scores
    axs[0].plot(k_values, metrics['ndcg'])
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
    
    # Plot MAP scores
    axs[1].plot(k_values, metrics['map'])
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
    
    plt.tight_layout()
    plt.show()

def plot_results_clustering(q_rel, Q):

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    query_url = "cranfield_embeddings\query_embeddings_svd.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_svd.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_svd.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    axs[0].plot(range(2, 10), results['ndcg'], label = "SVD")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map'], label = "SVD")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_embeddings\query_embeddings_word2vec.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_word2vec.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_word2vec.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    axs[0].plot(range(2, 10), results['ndcg'], label = "W2V")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map'], label = "W2V")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_sequences\q_seq_embedding.csv"
    docs_url = "cranfield_sequences\doc_seq_embedding.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_seq.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    axs[0].plot(range(2, 10), results['ndcg'], label = "S_1")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map'], label = "S_1")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')

    query_url = "cranfield_sequences\q_seq_embedding_2.csv"
    docs_url = "cranfield_sequences\doc_seq_embedding_2.csv"
    centroid_url = "pickel_dictionaries\centroid_mapping_seq_2.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    centroids = pickle.load(open(centroid_url, "rb"))

    q_ret = gen_result_centroids(queries , docs, centroids)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    axs[0].plot(range(2, 10), results['ndcg'], label = "S_2")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map'], label = "S_2")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
        
    plt.tight_layout()
    plt.legend()

    query_url = "cranfield_embeddings\query_embeddings_vs.csv"
    docs_url = "cranfield_embeddings\doc_embeddings_vs.csv"
    inverted_index_url = "pickel_dictionaries\inverted_index.pkl"

    queries = np.genfromtxt(query_url, dtype=float)
    docs = np.genfromtxt(docs_url, dtype=float)
    inverted_index = pickle.load(open(inverted_index_url, "rb"))

    q_ret = gen_result_index(queries , docs, Q, inverted_index)
    results = calculate_metrics(q_ret, q_rel, range(2,10))

    axs[0].plot(range(2, 10), results['ndcg'], label = "Vector Space", linestyle = '--')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('NDCG')
    axs[0].set_title('NDCG@k')
        
    axs[1].plot(range(2, 10), results['map'], label = "Vector Space", linestyle = '--')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('MAP')
    axs[1].set_title('MAP@k')
        
    plt.title("Using clustering for retrieving documents")
    plt.tight_layout()
    plt.legend()