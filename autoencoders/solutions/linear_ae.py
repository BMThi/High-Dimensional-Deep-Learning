class LinearAutoencoder(nn.Module):
    def __init__(self, hidden_dim, width, depth):
        super(LinearAutoencoder, self).__init__()
        
        # Intialize the encoder and decoder as empty Sequential models
        # We will add layers dynamically to these Sequential models according to the depth
        encoder = nn.Sequential()
        decoder = nn.Sequential()

        if depth < 1:
            raise ValueError("Depth must be at least 1") # TODO: Raise a ValueError

        if depth == 1:
            # In this case, we ignore the width parameter
            # and add a single linear layer to the encoder and decoder
            # with the appropriate dimensions.
            encoder.add_module("linear_1", nn.Linear(28*28, hidden_dim))
            decoder.add_module("linear_1", nn.Linear(hidden_dim, 28*28))

        else:
            encoder.add_module("linear_1", nn.Linear(28*28, width))
            for i in range(2, depth):
                encoder.add_module(f"linear_{i}", nn.Linear(width, width))
            encoder.add_module(f"linear_{depth}", nn.Linear(width, hidden_dim))

            decoder.add_module("linear_1", nn.Linear(hidden_dim, width))
            for i in range(2, depth):
                decoder.add_module(f"linear_{i}", nn.Linear(width, width))
            decoder.add_module(f"linear_{depth}", nn.Linear(width, 28*28))

        # Assign the encoder and decoder to self so that they are recognized as part of the model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # Encode the input
        x = self.encoder(x)
        # Decode the latent representation
        x = self.decoder(x)
        return x