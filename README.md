# NLP Intent Analysis

This notebook studies NLP in the context of Intent Analysis. The aim is to build a classifier to identify the user intent from a labelled set of user inquiries. 
A multiclass LSTM model with softmax output layer is used to identify the probabilities of a given input belonging to each of 16 possible outcome labels from the  
[ATIS aitline](https://www.kaggle.com/siddhadev/ms-cntk-atis) dataset.

The model, code, approach and confusion matrix analysis are detailed in the notebook [here](https://github.com/dstarkey23/NLP_bert_test/blob/master/NLP_Intent.ipynb).

# Word Embeddings
Uses GloVe non-contextual word embeddings to represent each word as a 100d vector.
The data set can be downloaded [here](http://nlp.stanford.edu/data/glove.6B.zip) or from a linux command line using
`wget http://nlp.stanford.edu/data/glove.6B.zip`.

