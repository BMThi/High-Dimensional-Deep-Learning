word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

print('Index of word "good":', word_to_idx['good'])
print('First word in the vocabulary:', idx_to_word[0])