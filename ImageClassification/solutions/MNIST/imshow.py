fig = plt.figure(figsize=(9, 5))

for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    sample_index = rd.choice( np.where(y_train==i)[0] )
    ax.imshow(x_train[sample_index], cmap=plt.cm.gray_r, interpolation='nearest')
    # ax.set_title("Label: %d" % y_train[sample_index])
    ax.set_title("Lab: %d | idx: %d" % (y_train[sample_index], sample_index))
    ax.grid(False)
    ax.axis('off')
        
plt.tight_layout()
plt.show()