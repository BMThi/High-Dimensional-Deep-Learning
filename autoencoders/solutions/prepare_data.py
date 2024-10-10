# Flatten the images
train_dataset = train_dataset.data.view(-1, 28*28).float()
train_dataset = train_dataset / 255.0
test_dataset = test_dataset.data.view(-1, 28*28).float()
test_dataset = test_dataset / 255.0