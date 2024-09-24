N_train, N_x_pixels, N_y_pixels = x_train.shape
N_test = x_test.shape[0]
N_classes = np.unique(y_train).shape[0]  #len(set(y_train))

print("Train data: %d images  (%d x %d pixels)" %(N_train, N_x_pixels, N_y_pixels))
print("Test data: %d images  (%d x %d pixels)" %(N_test, N_x_pixels, N_y_pixels))
print("Number of classes: %d classes" %N_classes)