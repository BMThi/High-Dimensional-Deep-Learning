batch_size = 256
epochs = 10

# Data normalization
x_train_flatten_norm = x_train.reshape((N_train, N_x_pixels*N_y_pixels))/255
x_test_flatten_norm = x_test.reshape((N_test, N_x_pixels*N_y_pixels))/255

# Network definition
mlp_norm = Sequential()
mlp_norm.add(Input(shape=(N_dim_flatten,)))
mlp_norm.add(Dense(128, activation='relu'))
mlp_norm.add(Dropout(0.2))
mlp_norm.add(Dense(128, activation='relu'))
mlp_norm.add(Dropout(0.2))
mlp_norm.add(Dense(N_classes, activation='softmax'))

# Network configuration
mlp_norm.compile(loss = 'sparse_categorical_crossentropy',
            optimizer = RMSprop(),
            metrics = ['accuracy'])

# Network training
t_train_mlp_norm = time.time()
history_norm = mlp_norm.fit(x_train_flatten_norm, y_train,
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose = 1,
                            validation_data = (x_test_flatten_norm, y_test))
t_train_mlp_norm = t_train_mlp_norm - time.time()

score_mlp_norm = mlp_norm.evaluate(x_test_flatten_norm, y_test, verbose=0)
predict_mlp_norm = mlp_norm.predict(x_test_flatten_norm)

# Results
print('Test loss:', score_mlp_norm[0])
print('Test accuracy:', score_mlp_norm[1])
print("Running time: %.2f seconds" %t_train_mlp_norm)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(1,1,1)
ax = sns.heatmap(pd.DataFrame(confusion_matrix(y_test, predict_mlp_norm.argmax(1))), annot=True, fmt="d")