score_mlp = mlp.evaluate(x_test_flatten, y_test, verbose=0)
predict_mlp = mlp.predict(x_test_flatten)

print('Test loss:', score_mlp[0])
print('Test accuracy:', score_mlp[1])
print("Running time: %.2f seconds" %t_train_mlp)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(1,1,1)
ax = sns.heatmap(pd.DataFrame(confusion_matrix(y_test, predict_mlp.argmax(1))), annot=True, fmt="d")