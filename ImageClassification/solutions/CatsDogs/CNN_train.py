epochs = 10

cnn_simple.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(learning_rate=3e-4),
    metrics = ['accuracy']
)

cnn_simple_augmented_history = model.fit(
    train_generator_augmented,
    validation_data = validation_generator,
    epochs = epochs
)