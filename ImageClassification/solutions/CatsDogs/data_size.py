total_train = train_df.shape[0]
total_validation = validation_df.shape[0]
total_test = test_df.shape[0]

print("We have %d training data, %d validation data and %d test data, both labels combined." % 
      (total_train, total_validation, total_test))