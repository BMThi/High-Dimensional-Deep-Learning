class TransformerBlock(layers.Layer):
    # embed_dim: Size of embeddings, maintained across the various layers
    # num_heads: number of heads in the attention layer
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        ## Definition of the different layers making up the block
        # Attention layer
        self.layerAttn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # First Normalization layer
        self.layerNorm1 = layers.LayerNormalization()
        # Dense layer (Feed-Forward)
        self.layerDense = layers.Dense(embed_dim, activation="relu")
        # Second Normalization layer
        self.layerNorm2 = layers.LayerNormalization()

    def call(self, inputs):
        ## Applying successive layers to inputs
        attn_output = self.layerAttn(inputs, inputs)
        norm_output1 = self.layerNorm1(inputs + attn_output)
        dense_output = self.layerDense(norm_output1)
        return self.layerNorm2(norm_output1 + dense_output)