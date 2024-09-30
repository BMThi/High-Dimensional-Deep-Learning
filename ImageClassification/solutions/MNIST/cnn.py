# Network definition
ConvNet = Sequential()
ConvNet.add(Input(shape=(28,28, 1)))
ConvNet.add(Conv2D(32, kernel_size=(3, 3), activation='relu', data_format="channels_last"))
ConvNet.add(Conv2D(64, (3, 3), activation='relu'))
ConvNet.add(MaxPooling2D(pool_size=(2, 2)))
ConvNet.add(Dropout(0.25))
ConvNet.add(Flatten())
ConvNet.add(Dense(128, activation='relu'))
ConvNet.add(Dropout(0.5))
ConvNet.add(Dense(N_classes, activation='softmax'))
ConvNet.summary()

# Network configuration
ConvNet.compile(loss = "sparse_categorical_crossentropy",
              optimizer = Adadelta(),
              metrics = ['accuracy'])

# Network training
t_train_ConvNet = time.time()
ConvNet.fit(x_train_conv, y_train,
          batch_size = batch_size,
          epochs = 10,
          verbose = 1,
          validation_data = (x_test_conv, y_test))
t_train_ConvNet = time.time() - t_train_ConvNet