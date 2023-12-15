embedding_size = 32

rnn = Sequential(name="RNN")
rnn.add(layers.Embedding(vocab_size, embedding_size, input_length=max_words))
rnn.add(layers.LSTM(int(.5*embedding_size)))
rnn.add(layers.Dropout(0.1))
rnn.add(layers.Dense(1, activation='sigmoid'))

print(rnn.summary())