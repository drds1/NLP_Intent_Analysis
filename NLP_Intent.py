#!/usr/bin/env python
# coding: utf-8

# # Intent Analysis on chat bot data
# 
# Chat bots are increasingly used to automate online customer queries and negate the need for call center staff.
# 
# Here we will train a Long Short Term Memory (LSTM) network to suggest the intent of a customer based on the text of the input query.
# 
# [LSTM](https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470) networks are great for text classification problems because they have memory (e.g. would be able to identify the word "not" before "good" as a negative sentiment). This sequential context is something that humans take for granted but that is very difficult for computers to grasp.
# 

# In[ ]:


#import dependencies
import pandas as pd
import keras
import numpy as np
import utils
import os
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import cosine_similarity


# if word embeddings exist pass else download them
word_embeddings_link = 'http://nlp.stanford.edu/data/glove.6B.zip'
word_embeddings_unzipped_file = 'glove.6B.zip'
word_embeddings_file = 'glove.6B.100d.txt'
word_embeddings_path = './data_nontracked'#../disaster_nlp/data/non_tracked/glove.6B.100d.txt'
if os.path.isdir(word_embeddings_path) is False:
    os.system('mkdir '+word_embeddings_path)
if os.path.isfile(word_embeddings_path+'/'+word_embeddings_file) is False:
    os.system('wget '+word_embeddings_link+' -P '+word_embeddings_path)
    os.system('unzip '+word_embeddings_path+'/'+word_embeddings_unzipped_file+' -d '+word_embeddings_path)

# # Load train and test sample
# 
# We will load separately the train and test data. The Xtrain and ytrain respectively will contain a list of the query (one list element per sample) and the outcome variable (10 possibilities).

# In[ ]:


#intent analysis atis dataset 
dftrain = pd.read_csv('./data/datasets_117486_281522_atis.train.csv')
Xtrain, ytrain = list(dftrain.values[:,1]), list(dftrain.values[:,-1])


dftest = pd.read_csv('./data/datasets_117486_281522_atis.test.csv')
Xtest, ytest = list(dftest.values[:,1]), list(dftest.values[:,-1])


# # One-hot encode labels
# 
# We will be predicting the probability that an input beloings to each class. For each label, we therefore need a vector of 1s and 0s where 1 appears next to the correct label, and all incorrect labels are zero.

# In[ ]:


#one hot encode labels
ytrain = pd.get_dummies(ytrain)
labels = list(ytrain.columns)
nlabels = len(labels)
ytrain = np.array(ytrain)
ytest = pd.get_dummies(ytest)
ytest = np.array(ytest)


# # Tokenization
# 
# We now map each word to an integer identifier. The list sample is then converted into a vector of integers corresponding to each word. For shorter inputs, we padd the start with zeros so that each sample has the same input vector length. 
# 
# Since there are so many words collectively in the full dataset, we keep only the 50 most common in each sample to avoid overfitting our classifier.

# In[ ]:


#convert all abstracts to sequences of integers key stored in idx_word
tokenizer = keras.preprocessing.text.Tokenizer(num_words=50,
                                               filters="?!':;,.#$&()*+-<=>@[\\]^_`{|}~\t\n",
                                               lower=True, split=' ')
tokenizer.fit_on_texts(Xtrain)
#assign an integer ID to each word
Xtrain_sequence = tokenizer.texts_to_sequences(Xtrain)
#padd the sequences of short sentences with 0s so everything is the same length
Xtrain_sequence = keras.preprocessing.sequence.pad_sequences(Xtrain_sequence,
                                                             padding='post')

#record the word to ID map and count the number of words in our vocabulary (+ 1 as we have the 0 padding as a word)
idx_word = tokenizer.index_word
num_words = len(idx_word) + 1


# # Word Embeddings
# 
# We need a mathematical way to represent 'words' in vector form such that words with similar meaning have vectors that point in similar directions. i.e. "plant" and "flower" would have similar pointing vectors in this abstract "Embedding Vector Space", but that vectors for words like "Hot" and "Cold" would point in opposite directions. Training these word embeddings is a herculean task for GPU's. Fortunately, other boffins have done the job for us and we can load a pre-trained word embedding dictionary. The 'glove' 100d word embeddings represents words in a 100 dimensional vector space. The more dimensions, the better linguistic understanding of our classifier, but the more compute time and sample size is needed.

# In[ ]:


#load in word embeddings
embeddings_dict = utils.load_embeddings(word_embeddings_path+'/'+word_embeddings_file)
embeddings_words = list(embeddings_dict.keys())
wordvec_dim = embeddings_dict[embeddings_words[0]].shape[0]
embedding_matrix = np.zeros((num_words,wordvec_dim))
for i, word in idx_word.items():
    # Look up the word embedding
    vector = embeddings_dict.get(word, None)
    # Record in matrix
    if vector is not None:
        embedding_matrix[i, :] = vector


# # Fit the model
# 
# We now build the LSTM model. Many neural nets share similar features. If we want to assign the probability to each class of a given input, the output will always be a Dense layer with softmax activation function equal to the number of labels. If we want a positive / negative decision, the output will be a single neuron with a sigmoid loss. The differences with LSTMs are the input 'Embedding' layer, where we specify our newly loaded word embeddings, and the number of input training samples and size of our vocabulary.

# In[ ]:


#Initialise the usual sequential network
model_lstm = keras.Sequential()

#initialise Ebedding layer
# input_length is the number of words ids per sample e.g 28
# NOT the sample size of the training data
# you do not need to supply that info
model_lstm.add(keras.layers.Embedding(input_dim=num_words,
                                      input_length=Xtrain_sequence.shape[1],
                                      output_dim=wordvec_dim,
                                      weights=[embedding_matrix],
                                      trainable=False,
                                      mask_zero=True))

#words which are not in the pretrained embeddings (with value 0) are ignored
model_lstm.add(keras.layers.Masking(mask_value = 0.0))

# Recurrent layer
model_lstm.add(keras.layers.LSTM(64, activation='relu'))

# Dropout for regularisation and avoid overfit
model_lstm.add(keras.layers.Dropout(0.5))

# Output layer
model_lstm.add(keras.layers.Dense(nlabels,activation = 'softmax' ))

# Compile the model
model_lstm.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summarise the model
model_lstm.summary()


# # Fit the model to the training data

# In[ ]:


model_lstm.fit(Xtrain_sequence, ytrain,epochs=5)


# # Transform the test data using the tokenizer and evaluate the model performance

# In[ ]:


#Transform the test data
Xtest_sequence = tokenizer.texts_to_sequences(Xtest)
Xtest_sequence = keras.preprocessing.sequence.pad_sequences(Xtest_sequence,
                                                            maxlen = Xtrain_sequence.shape[1],
                                                            padding='post')


# In[ ]:


#evaluate model performance
eval_lstm = model_lstm.evaluate(Xtest_sequence, ytest)
ypred = model_lstm.predict(Xtest_sequence)

#multiclass problem, pick highest probability as the choice
ypred_choices = ypred*0
cnt = 0
for yp in ypred:
    idx = np.argmax(yp)
    ypred_choices[cnt,idx] = 1
    cnt += 1

#construct multilabel confusion matrix
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
cm = metrics.confusion_matrix(ytest.argmax(axis=1), ypred_choices.argmax(axis=1))
cmnorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
plt.rcParams.update({'font.size': 7})
# NOTE: Fill all variables here with default values of the plot_confusion_matrix
disp = disp.plot(xticks_rotation=90,cmap='Blues')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

#Now make a plot showing the F1 score (harmonic mean precision recall)
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis = 0) - true_pos
false_neg = np.sum(cm, axis = 1) - true_pos
f1 = true_pos / (true_pos + 0.5*(false_pos + false_neg))

fig = plt.figure()
ax1 = fig.add_subplot(111)
idxsort = np.argsort(f1)[-1::-1]
x = np.arange(len(labels))
y = f1[idxsort]
ax1.bar(x,y,color='b')
ax1.set_xticks(x)
ax1.set_xticklabels(np.array(labels)[idxsort])
ax1.set_ylabel('F1',color='b')
ax1.spines['left'].set_color = 'blue'
ax1.set_title('F1 metric (precision, recall harmonic mean)')
ax1.tick_params(axis='x',rotation=90)
ax1.tick_params(axis='y',colors='blue')
#overplot the label counts on a second axis
ax2 = ax1.twinx()
counts = np.sum(cm,axis=1)
ax2.plot(x, counts[idxsort],color='r',label='label counts')
ax2.spines['right'].set_color = 'red'
ax2.set_ylabel('Label Counts',color='r')
ax2.tick_params(axis='y',colors='red')
plt.tight_layout()
plt.savefig('F1_score.png')

