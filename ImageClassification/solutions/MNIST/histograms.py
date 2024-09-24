# Seaborn

plt.figure()
sns.histplot(y_train, stat='proportion', discrete=True, alpha=.8, shrink=.8, label='Train set')
sns.histplot(y_test, stat='proportion', discrete=True, alpha=.5, shrink=.8, label='Test set')

plt.title('Number distribution in test and train sets')
plt.legend()
plt.show()


# --- #
# Matplotlib

plt.figure()
plt.hist(y_train, density=True, alpha=0.6, label='train set')
plt.hist(y_test, density=True, alpha=0.4, label='test set')

plt.title('Number distribution in test and train sets')
plt.legend()
plt.show()