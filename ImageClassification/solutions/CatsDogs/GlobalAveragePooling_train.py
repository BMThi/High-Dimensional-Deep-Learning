# Training the last dense layers of the network

epochs = 10
conv_base.trainable = False

vgg_combined_average.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(learning_rate=3e-4),
    metrics = ['accuracy']
)

history_combined_average = vgg_combined_average.fit(
    train_generator_augmented,
    epochs = epochs,
    validation_data = validation_generator)