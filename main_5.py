import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Callable, Union
import seaborn as sns

# --- Configuration ---
# !!! IMPORTANT !!!
# Change this to your actual student ID
STUDENT_ID = "r119020XX" 

# --- DO NOT CHANGE THESE ---
PUBLIC_DATA_PATH = "/Users/zhangyijie/Documents/ 113-2 BDA/final_project/big data 2/public_data.csv"
PRIVATE_DATA_PATH = "/Users/zhangyijie/Documents/ 113-2 BDA/final_project/big data 2/private_data.csv"
PUBLIC_NUM_CLUSTERS = 15
PRIVATE_NUM_CLUSTERS = 23

def create_submission(df, labels, student_id, dataset_type):
    """Creates a submission DataFrame in the required format."""
    # Ensure labels are integers, which is the expected format
    submission_df = pd.DataFrame({'id': df.index, 'label': labels.astype(int)})
    return submission_df

def visualize_tsne_clusters(tsne_results, labels, title):
    """(Optional) Creates a scatter plot to visualize the t-SNE results."""
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1], 
        hue=labels, 
        palette=sns.color_palette("viridis", n_colors=len(np.unique(labels))), 
        legend='full'
    )
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def final_clustering_pipeline(features, n_clusters, perplexity=40, random_state=42):
    """
    Runs the final dimensionality reduction and clustering step.
    This pipeline first applies t-SNE to create a 2D representation, then
    clusters this representation using AgglomerativeClustering.
    """
    if len(features) <= n_clusters:
        # Handle cases with very few data points
        return np.arange(len(features))

    print("  - Applying final t-SNE for clustering...")
    # Increased n_iter for better convergence, essential for stable results.
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1500, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    
    print("  - Clustering final t-SNE results with AgglomerativeClustering...")
    # 'ward' linkage is a strong default as it minimizes variance within clusters.
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg_cluster.fit_predict(tsne_results)
    
    # You can uncomment the line below to see the final t-SNE plot!
    # visualize_tsne_clusters(tsne_results, labels, f'Final t-SNE with {n_clusters} Clusters')
    
    return labels


def final_clustering_pipeline_simple(features, n_clusters):
    """
    Runs a simplified final clustering pipeline using KMeans.
    This is a fallback for cases where AgglomerativeClustering is not suitable.
    """
    if len(features) <= n_clusters:
        return np.arange(len(features))

    # print("  - Applying KMeans for final clustering...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(features)

    print("  - Clustering with AgglomerativeClustering...")
    # agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    # labels = agg.fit_predict(features)
    
    return labels

def preprocess_data(df):
    """
    Preprocesses the data using a log transform and a robust scaler.
    This simplifies the previous two-step scaling process. RobustScaler is
    less sensitive to outliers than MinMaxScaler.
    """
    # Using np.log1p to handle potential skewness in the data.

    # first divide the value of its maximum value in the column
    # to ensure all values are positive before applying log transformation.
    # This is a common practice in data preprocessing to stabilize variance.
    df_copy = df.copy()
    
    # Apply log1p to handle skewness
    # df_copy = np.log1p(df_copy)


    for col in df_copy.columns:
        
        col_99 = df_copy[col].quantile(0.995)
        col_5 = df_copy[col].quantile(0.05)
        # # clip values to the 5th and 95th percentiles
        df_copy[col] = df_copy[col].clip(upper=col_99)

        col_max = df_copy[col].max()

        if col_max > 0:
            df_copy[col] = df_copy[col] / col_max
        else:
            df_copy[col] = 0


    


    scaled_features = df_copy.astype(float).values
    
    return scaled_features


def make_cluster_onehots(
    df: pd.DataFrame,
    cols: list[int],
    n_clusters: int,
    scaler_func: Callable[[pd.DataFrame], np.ndarray] = preprocess_data,
    cluster_model: object = None
) -> np.ndarray:
    """
    1) Select df[cols],
    2) scale via scaler_func,
    3) cluster into n_clusters,
    4) one-hot encode the labels,
    5) return the one-hot matrix.
    """
    # 1. slice out the two (or more) dimensions
    sub = df.iloc[:, cols] if isinstance(cols[0], int) else df.loc[:, cols]

    # 2. scale them
    X = scaler_func(sub)

    # 3. cluster
    if cluster_model is None:
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = cluster_model.fit_predict(X)

    # 4. one-hot encode
    enc = OneHotEncoder(sparse_output=False)
    onehots = enc.fit_transform(labels.reshape(-1, 1))
    return onehots

# def feature_engineering_pipeline(df, n_total_clusters):
#     """
#     A refined feature engineering pipeline based on the project hint.
    
#     1.  It first performs a preliminary clustering on dimensions 2 and 3 to
#         find the 5 "obvious" clusters mentioned in the hint.
#     2.  The labels from this clustering are one-hot encoded to create
#         powerful new features.
#     3.  These new features are combined with the scaled original data.
#     4.  The final clustering is performed on this augmented feature set.
#     """
#     print("Starting refined feature engineering pipeline...")
    
#     # --- Step 1: Preliminary Clustering on Dimensions 2 & 3 ---
#     print("Step 1: Performing preliminary clustering on dimensions 2 & 3...")
#     features_s2_s3 = df.iloc[:, [1, 2]] # Select columns 'S2' and 'S3'
    
#     # Preprocess only these two dimensions
#     scaled_features_s2_s3 = preprocess_data(features_s2_s3)
    
#     # Cluster these 2D points into 5 groups as per the hint
#     # Using AgglomerativeClustering as it's good for non-spherical shapes
#     # preliminary_cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
#     preliminary_cluster = KMeans(n_clusters=5, init='k-means++', random_state=42)
#     preliminary_labels = preliminary_cluster.fit_predict(scaled_features_s2_s3)
    
#     # --- Step 2: Create New Features from Preliminary Labels ---
#     print("Step 2: One-hot encoding preliminary cluster labels...")
#     encoder = OneHotEncoder(sparse_output=False)
#     # Reshape labels to be a column vector for the encoder
#     one_hot_encoded_labels = encoder.fit_transform(preliminary_labels.reshape(-1, 1))

#     # --- Step 3: Preprocess the Full Original Dataset ---
#     print("Step 3: Preprocessing the full original dataset...")
#     scaled_original_features = preprocess_data(df)

#     # --- Step 4: Combine Features ---
#     print("Step 4: Combining scaled original data with new one-hot encoded features...")
#     # This creates a rich feature set for the final clustering step
#     augmented_features = np.concatenate([
#         scaled_original_features, 
#         one_hot_encoded_labels
#     ], axis=1)

#     # --- Step 5: Run the Final Clustering Pipeline ---
#     print(f"Step 5: Running final clustering to find {n_total_clusters} clusters...")
#     final_labels = final_clustering_pipeline_simple(augmented_features, n_total_clusters)
    
#     return final_labels
def feature_engineering_pipeline(
    df: pd.DataFrame,
    n_total_clusters: int,
    axis_list: list[list[int]],
    prelim_clusters: Union[int, list[int]] = 5,
):
    """
    df: your full DataFrame
    axis_list: list of column‐index pairs, e.g. [[1,2], [3,4], …]
    prelim_clusters: either a single int, or a list of ints (same length as axis_list)
    """
    print("Starting refined feature engineering pipeline...")

    all_onehots = []
    for i, axes in enumerate(axis_list):
        # decide how many preliminary clusters for this pair
        if isinstance(prelim_clusters, int):
            k = prelim_clusters
        else:
            k = prelim_clusters[i]
        print(f"  - clustering on axes {axes} → {k} clusters")
        
        oh = make_cluster_onehots(
            df,
            cols=axes,
            n_clusters=k
        )
        all_onehots.append(oh)

    # stack all preliminary one-hot features
    multi_onehots = np.concatenate(all_onehots, axis=1)

    # preprocess the full data
    print("  - preprocessing full dataset")
    original_scaled = preprocess_data(df)

    # concatenate original + all one-hots
    augmented = np.concatenate([original_scaled, multi_onehots], axis=1)

    # final clustering
    print(f"  - final clustering into {n_total_clusters} clusters")
    final_labels = final_clustering_pipeline_simple(augmented, n_total_clusters)
    return final_labels

def main():
    """Main function to run the clustering process."""
    print("Starting refined clustering process...")

    # --- Process the Public Dataset ---
    print("\nProcessing public dataset...")
    try:
        # Load data, assuming first column is an ID to be used as index
        public_df = pd.read_csv(PUBLIC_DATA_PATH, index_col=0)
        public_labels = feature_engineering_pipeline(public_df, PUBLIC_NUM_CLUSTERS, axis_list=[[1,2],[0,1]],prelim_clusters=[ 5,4])
        
        # Create submission file using the original index
        public_submission = create_submission(public_df, public_labels, STUDENT_ID, 'public')
        public_filename = f"{STUDENT_ID}_public.csv"
        
        # The official eval script expects 'public_submission.csv'
        public_submission.to_csv("public_submission.csv", index=False)
        print(f"Public label distribution: {np.bincount(public_labels)}")
        print(f"Successfully created public submission file: public_submission.csv")
        print(f"Also saved as: {public_filename}")

    except Exception as e:
        print(f"\nAn error occurred while processing the public data: {e}")
        import traceback
        traceback.print_exc()

    # --- Process the Private Dataset ---
    print("\nProcessing private dataset...")
    try:
        # Load data, assuming first column is an ID to be used as index
        private_df = pd.read_csv(PRIVATE_DATA_PATH, index_col=0)
        private_labels = feature_engineering_pipeline(private_df, PRIVATE_NUM_CLUSTERS, axis_list=[[1,2],[0,1]],prelim_clusters=[ 5,4])
        
        private_submission = create_submission(private_df, private_labels, STUDENT_ID, 'private')
        private_filename = f"{STUDENT_ID}_private.csv"
        private_submission.to_csv("private_submission.csv", index=False)
        print(f"Private label distribution: {np.bincount(private_labels)}")
        print(f"Successfully created private submission file: {private_filename}")

    except Exception as e:
        print(f"\nAn error occurred while processing the private data: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nProcess finished.")

if __name__ == '__main__':
    main()