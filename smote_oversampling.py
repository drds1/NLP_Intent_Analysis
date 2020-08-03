import pandas as pd
import keras
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity



def load_embeddings(file):
    '''
    load embeddings txt file into a dictionary
    :param file:
    :return:
    '''
    embeddings_dictionary = dict()
    glove_file = open(file, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    return embeddings_dictionary


dftrain = pd.read_csv('./data/datasets_117486_281522_atis.train.csv')
Xtrain, ytrain = list(dftrain.values[:,1]), list(dftrain.values[:,-1])
ytrain = pd.get_dummies(ytrain)
labels = list(ytrain.columns)
nlabels = len(labels)
ytrain = np.array(ytrain)
dftest = pd.read_csv('./data/datasets_117486_281522_atis.test.csv')
Xtest, ytest = list(dftest.values[:,1]), list(dftest.values[:,-1])
ytest = pd.get_dummies(ytest)
ytest = np.array(ytest)

#convert all abstracts to sequences of integers key stored in idx_word
tokenizer = keras.preprocessing.text.Tokenizer(num_words=50,
                                               filters="?!':;,.#$&()*+-<=>@[\\]^_`{|}~\t\n",
                                               lower=True, split=' ')
tokenizer.fit_on_texts(Xtrain)
#assign number to each word
Xtrain_sequence = tokenizer.texts_to_sequences(Xtrain)
#padd the sequences of short sentences with 0s so everything is the same length
Xtrain_sequence = keras.preprocessing.sequence.pad_sequences(Xtrain_sequence,
                                                             padding='post')

idx_word = tokenizer.index_word
num_words = len(idx_word) + 1


#load in word embeddings
embeddings_dict = load_embeddings('../disaster_nlp/data/non_tracked/glove.6B.100d.txt')
#embeddings_df = pd.DataFrame(embeddings_dict).transpose()
#embeddings_df_cosine_similarity = cosine_similarity(embeddings_df)
#embeddings_df['neural'].corrwith(embeddings_df)
embeddings_words = list(embeddings_dict.keys())
wordvec_dim = embeddings_dict[embeddings_words[0]].shape[0]
embedding_matrix = np.zeros((num_words,wordvec_dim))
for i, word in enumerate(idx_word.keys()):
    # Look up the word embedding
    vector = embeddings_dict.get(word, None)
    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector


#wordvec_dim = 100
model_lstm = keras.Sequential()
#initialise Ebedding layer num_words = len(idx_word) + 1 to deal with 0 padding
#input_length is the number of words ids per sample e.g 28
# NOT the sample size of the training data
# you do not need to supply that info
model_lstm.add(keras.layers.Embedding(input_dim=num_words,
                                      input_length=Xtrain_sequence.shape[1],
                                      output_dim=wordvec_dim,
                                      weights=[embedding_matrix],
                                      trainable=False,
                                      mask_zero=True))

embedding_test = model_lstm.predict(Xtrain_sequence)


#isolate examples belonging to the underrepresented class
classnum = ytrain.sum(axis=0)
idxmin = np.argmin(classnum)
idx_minclass = np.where(ytrain[:,idxmin] == 1)[0]
label_minclass = labels[idxmin]
Xtrain_minclass = [Xtrain[i] for i in idx_minclass]
Xtrain_sequence_minclass = Xtrain_sequence[idx_minclass,:]
print('the most underrepresented class is: "'+label_minclass+'"')
print('with '+str(len(Xtrain_minclass)),'entries')
for xt in Xtrain_minclass:
    print(xt)
