test_prediction = vgg_combined.predict(test_generator)
score_test = vgg_combined.evaluate(test_generator)
print('Train accuracy:', score_test[1])

# --- #

fig = plt.figure(figsize=(10,10))

test_imgs_idx = np.random.randint(low=0, high=test_df.shape[0], size=(9,))

for i, idx in enumerate(test_imgs_idx):
    img = img_to_array( load_img(path + "test/" + test_df['filename'][idx]) )/255
    pred = test_prediction[idx][0]
    
    ax = fig.add_subplot(3,3,i+1)
    ax.imshow(img, interpolation='nearest')
    color = "green"
    if pred >0.5:
        title = "Probabiliy for dog : %.1f" % (pred*100)
        if test_df['category'][idx] == '0':
            color = "red"
    else:
        title = "Probabiliy for cat : %.1f" %((1-pred)*100)
        if test_df['category'][idx] == '1':
            color = "red"
    ax.set_title(title, color=color)

plt.tight_layout()
plt.show()