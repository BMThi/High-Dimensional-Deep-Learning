class TokenAndPositionEmbedding(Layer):
    def __init__(self, max_words, vocab_size, embed_dim):
        super().__init__()
        ## Definition of the Embedding block layers : Token AND Position     
        # Token Embedding
        self.tokenEmb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # Position Embedding
        self.posEmb = Embedding(input_dim=max_words, output_dim=embed_dim)

    def call(self, x):
        ## Embedding computation from input x (a sentence)
        # max_words: Sequence length
        # positions: All possible positions in the sequence
        max_words = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_words, delta=1)
        positions = self.posEmb(positions)
        x = self.tokenEmb(x)
        return x + positions