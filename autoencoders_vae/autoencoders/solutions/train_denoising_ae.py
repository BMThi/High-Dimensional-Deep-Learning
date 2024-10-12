# Train the denoising AE with different noise levels

# Set the hyper-parameters
noise_levels = [0.003, 0.01, 0.03, 0.1]  # Noise levels to test

# Store the models, along with the losses
denoising_models = {}
denoising_train_losses = {}
denoising_val_losses = {}
denoising_test_losses = {}

for p in HIDDEN_DIMS:
    for noise_level in noise_levels:
        print(f"Training denoising AE with {p} hidden dims and noise level {noise_level}")
        # Instantiate the model and move it to the device
        denoising_ae = Autoencoder(hidden_dim=p).to(device)

        # Set the loss and optimizer
        criterion = nn.MSELoss() # Mean Squared Error loss
        optimizer = optim.Adam(denoising_ae.parameters(), lr=1e-3) # Adam optimizer

        for epoch in range(EPOCHS):
            # Set the model to training mode
            denoising_ae.train()
            # Initialize running training and validation loss for the epoch
            running_train_loss = 0.0
            running_val_loss = 0.0

            # Training loop keeping in mind to:
            # generate noisy images with the given noise level
            # clip pixel values to [0, 1] after adding noise
            # do the forward pass on the noisy images
            # compute the loss by comparing the outputs with the ORIGINAL images
            for images in train_loader:
                # Move images to device
                images = images.to(device) 
                # Generate noisy images with the given noise level
                noisy_images = images + noise_level * torch.randn_like(images)
                # Clip pixel values to [0, 1]
                noisy_images = torch.clamp(noisy_images, 0, 1) 
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = denoising_ae(noisy_images)
                # Compute loss
                loss = criterion(outputs, images)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                # Accumulate training loss
                running_train_loss += loss.item() * images.size(0)
            
            # Compute average training loss for the epoch
            train_loss = running_train_loss / len(train_loader.dataset)

            # Validation loop keeping in mind to:
            # follow the same procedure as for the training loop.
            denoising_ae.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation for validation
                for images in val_loader:
                    # Move images to device
                    images = images.to(device)
                    # Generate noisy images with the given noise level
                    noisy_images = images + noise_level * torch.randn_like(images)
                    # Clip pixel values to [0, 1]
                    noisy_images = torch.clamp(noisy_images, 0, 1) 
                    # Forward pass
                    outputs = denoising_ae(noisy_images)
                    # Compute loss
                    loss = criterion(outputs, images)
                    # Accumulate validation loss
                    running_val_loss += loss.item() * images.size(0)

            # Compute average validation loss for the epoch
            val_loss = running_val_loss / len(val_loader.dataset)

            # Print training and validation statistics for the epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save model and losses
        denoising_models[(p, noise_level)] = denoising_ae
        denoising_train_losses[(p, noise_level)] = train_loss
        denoising_val_losses[(p, noise_level)] = val_loss

        # Evaluate the model on the test set
        denoising_ae.eval()  # Set model to evaluation mode
        running_test_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for images in test_loader:
                # Move images to device
                images = images.to(device)
                # Generate noisy images with the given noise level
                noisy_images = images + noise_level * torch.randn_like(images)
                # Clip pixel values to [0, 1]
                noisy_images = torch.clamp(noisy_images, 0, 1)
                # Forward pass
                outputs = denoising_ae(noisy_images)
                # Compute loss
                loss = criterion(outputs, images)
                # Accumulate test loss
                running_test_loss += loss.item() * images.size(0)
        
        # Compute and save the average test loss
        test_loss = running_test_loss / len(test_loader.dataset)
        denoising_test_losses[(p, noise_level)] = test_loss
        