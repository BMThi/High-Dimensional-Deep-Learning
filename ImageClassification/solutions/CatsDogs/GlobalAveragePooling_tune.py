# Final fine-tuning pass

epochs = 10
conv_base.trainable = True

model.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(learning_rate=1e-5), # Low learning rate
    metrics = ['accuracy']
)

history = model.fit(
    train_generator_augmented,
    epochs = epochs,
    validation_data = validation_generator)