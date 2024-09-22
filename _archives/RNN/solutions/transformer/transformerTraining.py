embed_dim = 32  # Embedding dimension for each word
num_heads = 1  # Number of attention heads

inputs = layers.Input(shape=(max_words,))
embedding_layer = TokenAndPositionEmbedding(max_words, vocab_size, embed_dim)
x = embedding_layer(inputs)

# "Succession" of transformers
transformer_block = TransformerBlock(embed_dim, num_heads)
x = transformer_block(x)
transformer_block2 = TransformerBlock(embed_dim, num_heads)
x = transformer_block2(x)

# Projection head : Global Average Pooling + Dense + ... 
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

# Model setup
transformer = tf.keras.Model(inputs=inputs, outputs=outputs)
transformer.summary()