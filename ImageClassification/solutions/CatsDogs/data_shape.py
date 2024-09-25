img_size = np.zeros((train_df.shape[0],2))
for i, filename in enumerate(train_df['filename']) :
    img = img_to_array( load_img(path + "train/train/" + filename) )
    img_size[i,:] = img.shape[:2]

# --- #

plt.figure()
ax = sns.boxplot(img_size)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(['width', 'height'])
plt.title("Image width and height")
plt.show()