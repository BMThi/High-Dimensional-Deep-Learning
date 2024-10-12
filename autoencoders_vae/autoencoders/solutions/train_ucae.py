# Train the undercomplete AE for the different hidden dimensions

# Store the models, along with the losses
ucae_models = {} # For visualization of image reconstructions
ucae_train_losses = {}
ucae_val_losses = {}
ucae_test_losses = {}

for p in HIDDEN_DIMS:
    print(f"Training undercomplete AE with {p} hidden dims")

    # Instantiate the model and move it to the device
    undercomplete_ae = Autoencoder(hidden_dim=p).to(device)

    # Set the loss and optimizer
    criterion = nn.MSELoss() # Mean Squared Error loss
    optimizer = optim.Adam(undercomplete_ae.parameters(), lr=1e-3) # Adam optimizer

    for epoch in range(EPOCHS):
        # Set the model to training mode
        undercomplete_ae.train()
        # Initialize running training and validation loss for the epoch
        running_train_loss = 0.0
        running_val_loss = 0.0

        # Training loop
        for images in train_loader:
            # Move images to device
            images = images.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = undercomplete_ae(images)
            # Compute loss
            loss = criterion(outputs, images)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Update running loss
            running_train_loss += loss.item() * images.size(0)
    
        # Compute average training loss for the epoch
        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation loop
        undercomplete_ae.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            for images in val_loader:
                # Move images to device
                images = images.to(device)
                # Forward pass
                outputs = undercomplete_ae(images)
                # Compute loss
                loss = criterion(outputs, images)
                # Accumulate validation loss
                running_val_loss += loss.item() * images.size(0)

        # Compute average validation loss for the epoch
        val_loss = running_val_loss / len(val_loader.dataset)

        # Print training and validation statistics for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Save the trained model and losses
    ucae_models[p] = undercomplete_ae
    ucae_train_losses[p] = train_loss
    ucae_val_losses[p] = val_loss

    # Evaluate the model on the test set
    undercomplete_ae.eval()  # Set model to evaluation mode
    running_test_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for images in test_loader:
            # Move images to device
            images = images.to(device)
            # Forward pass
            outputs = undercomplete_ae(images)
            # Compute loss
            loss = criterion(outputs, images)
            # Accumulate test loss
            running_test_loss += loss.item() * images.size(0)
    
    # Compute average test loss
    test_loss = running_test_loss / len(test_loader.dataset)
    ucae_test_losses[p] = test_loss