def train_pca(p):
    """Trains a PCA model with p principal components

    Args:
        p (int): Number of principal components
    
    Returns:
        PCA: Trained PCA model
    """

    pca = PCA(n_components=p)
    pca.fit(train_dataset)
    return pca