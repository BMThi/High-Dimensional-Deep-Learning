# Introduction to Recurrent Neural Networks **&** Transformers
_Sentiment analysis through Recurrent Neural Networks & Transformers_

---

In this tutorial, we are interested in the problem of sentiment analysis. 
* In the first part, we build a recurrent network on a toy dataset, starting from scratch, to determine whether a sentence is positive or negative. 
* Second, using the [`Keras`](https://keras.io/) API, we build several recurrent networks to determine whether a movie review is positive or negative.
* Finally, we propose a simple Transformer, again to judge whether the aforementioned reviews are positive or negative. 

The toy dataset is available in the file [data.py](data.py). For film reviews, we will use the [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/) database.