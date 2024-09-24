# Network definition
mlp = Sequential()
mlp.add(Input(shape=(N_dim_flatten,)))
mlp.add(Dense(128, activation='relu'))
mlp.add(Dropout(0.2))
mlp.add(Dense(128, activation='relu'))
mlp.add(Dropout(0.2))
mlp.add(Dense(N_classes, activation='softmax'))

# Summary
mlp.summary()