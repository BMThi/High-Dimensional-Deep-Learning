class RNN:
    # A Vanilla Recurrent Neural Network.

    def __init__(self, input_size, output_size, hidden_size=64):
        # Weights
        self.Whh = rd.randn(hidden_size, hidden_size) / 1000
        self.Wxh = rd.randn(hidden_size, input_size) / 1000
        self.Why = rd.randn(output_size, hidden_size) / 1000

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    # ----- #

    def forward(self, inputs):
        '''
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.
        - inputs is a list of one-hot vectors with shape (input_size, 1).
        '''
        h = np.zeros((self.Whh.shape[0], 1))

        # Perform each step of the RNN
        for x in inputs:
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

        # Compute the output
        y = self.Why @ h + self.by

        return y, h