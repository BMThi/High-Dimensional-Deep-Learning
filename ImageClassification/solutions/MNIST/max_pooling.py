def build_pool_layer(pool_size=(2,2)):    
    def my_init_filter(shape, conv_filter=conv_filter, dtype=None, partition_info=None):
        xf,yf = conv_filter.shape
        array = conv_filter.reshape(xf, yf, 1, 1)
        return array
    
    pool_layer = Sequential()
    pool_layer.add( Input(shape=(28, 28, 1)) )
    pool_layer.add( MaxPool2D(pool_size=pool_size) )
    return pool_layer


# Image choice : Digit or Plus
idx = sample_index[9] #####
x = x_train_conv[idx] #####
# x = img_plus #####

pool_layer = build_pool_layer()

# --- #

img_in = np.expand_dims(x, 0)
img_out = pool_layer.predict(img_in)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(img_in[0,:,:,0].astype(np.uint8), cmap="binary");
ax0.set_title("Original image")
ax0.grid(False)

ax1.imshow(img_out[0,:,:,0].astype(np.uint8), cmap="binary");
ax1.set_title("Pooled image")
ax1.grid(False)