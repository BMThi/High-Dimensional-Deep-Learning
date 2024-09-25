# Model definition

conv_base = VGG16(
    weights = 'imagenet',
    include_top = False,
    input_shape = (img_width, img_height, 3)
)

vgg_combined_average = Sequential()
vgg_combined_average.add(Input(shape=(img_width, img_height, 3)))
vgg_combined_average.add(conv_base)
vgg_combined_average.add(GlobalAveragePooling2D())
vgg_combined_average.add(Dense(256, activation='relu'))
vgg_combined_average.add(Dense(1, activation='sigmoid'))

vgg_combined_average.summary()