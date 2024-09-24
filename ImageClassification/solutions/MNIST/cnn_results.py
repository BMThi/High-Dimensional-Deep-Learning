score_ConvNet = ConvNet.evaluate(x_test_conv, y_test, verbose=0)
predict_ConvNet = ConvNet.predict(x_test_conv)

print('Test loss:', score_ConvNet[0])
print('Test accuracy:', score_ConvNet[1])
print("Time Running: %.2f seconds" %t_train_ConvNet )

fig=plt.figure(figsize=(7,6))
ax = fig.add_subplot(1,1,1)
ax = sns.heatmap(pd.DataFrame(confusion_matrix(y_test, predict_ConvNet.argmax(1))), annot=True, fmt="d")