def run_benchmark(hidden_dims, ae_widths, ae_depths):
    """Runs a benchmark on the given hidden dimensions, autoencoder widths, and autoencoder depths

    Args:
        hidden_dims (list): A list of hidden dimensions / principal components to test
        ae_widths (list): A list of autoencoder widths to test
        ae_depths (list): A list of autoencoder depths to test

    Returns:
        dict: Dictionary containing the PCA training loss
        dict: Dictionary containing the PCA test loss
        dict: Dictionary containing the autoencoder training loss
        dict: Dictionary containing the autoencoder test loss
    """
    
    # Initialize dictionaries to store results
    pca_train_loss = {}
    pca_test_loss = {}
    ae_train_loss = {}
    ae_test_loss = {}

    # Iterate through the hidden dimensions
    for p in hidden_dims:
        # Train the PCA model
        print(f"Training PCA model with {p} hidden dims")

        # Train a PCA model with p hidden dimensions
        pca = train_pca(p)

        # Build reconstruction of train/test data using the PCA model
        pca_reconstruction_train = pca.inverse_transform(pca.transform(train_dataset))
        pca_reconstruction_test = pca.inverse_transform(pca.transform(test_dataset))

        # Calculate the train / test MSE of the PCA model
        mse_train_pca = mean_squared_error(train_dataset, pca_reconstruction_train) # TODO: Evaluate the PCA model on train data
        mse_test_pca = mean_squared_error(test_dataset, pca_reconstruction_test) # TODO: Evaluate the PCA model on validation data
        print(f"Train MSE: {mse_train_pca}, Test MSE: {mse_test_pca}")
        
        # Store the results in the dictionaries
        pca_train_loss[p] = mse_train_pca
        pca_test_loss[p] = mse_test_pca

        # Iterate through the autoencoder widths and depths
        for width in ae_widths:
            for depth in ae_depths:
                print(f"Training autoencoder with {p} hidden dims, width={width}, depth={depth}")

                # Initialize and train the autoencoder model
                model, train_loss = train_autoencoder(hidden_dim=p, width=width, depth=depth)

                # Build reconstruction of test data using AE and compute test loss
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for images in test_loader:
                        images = images.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, images)
                        test_loss += loss.item() * images.size(0)
                    
                    test_loss /= len(test_loader.dataset)
                
                # Store the results in the dictionaries
                ae_train_loss[(p, width, depth)] = train_loss
                ae_test_loss[(p, width, depth)] = test_loss
                print(f"Train MSE: {train_loss}, Test MSE: {test_loss}")

    return pca_train_loss, pca_test_loss, ae_train_loss, ae_test_loss