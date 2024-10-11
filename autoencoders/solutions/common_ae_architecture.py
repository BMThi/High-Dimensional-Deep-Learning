class Autoencoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder: Linear layer with ReLU activation
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, hidden_dim),
            nn.ReLU()
        )

        # Decoder: Linear layer with Sigmoid activation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, WIDTH),
            nn.ReLU(),
            nn.Linear(WIDTH, INPUT_DIM),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode the input
        x = self.encoder(x)
        # Decode the latent representation
        x = self.decoder(x)
        return x