hidden_dims = [1, 2, 4, 8, 16]
ae_widths = [64] # We will only test one width, otherwise it takes too long...
ae_depths = [1, 2, 3]

# Run the benchmark
pca_train_loss, pca_test_loss, ae_train_loss, ae_test_loss = run_benchmark(hidden_dims, ae_widths, ae_depths)