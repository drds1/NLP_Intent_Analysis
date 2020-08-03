import pandas as pd
import keras
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity



def stop_word_removal(xlist,words=['is', 'the', 'of', 'a']):
    xl2 = []
    for xl in xlist:
        for w in words:
            xl = xl.replace(' '+w+' ',' ')
        xl2.append(xl)
    return xl2


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


class lstm_fit():

    def __init__(self):
        self.x = None
        self.word_embeddings = None
        self.tokenizer = None
        self.Xtrain = None
        self.Xtest = None
        self.ytest = None
        self.ytrain = None
        self.embedding_matrix = None
        self.model_lstm = None

    def tokenize(self):
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=100,
                                                       filters="?!':;,.#$&()*+-<=>@[\\]^_`{|}~\t\n",
                                                       lower=True, split=' ')

        tokenizer.fit_on_texts(self.Xtrain)
        self.tokenizer = tokenizer
        # assign number to each word
        self.Xtrain_sequence = tokenizer.texts_to_sequences(self.Xtrain)
        # padd the sequences of short sentences with 0s so everything is the same length
        self.Xtrain_sequence = keras.preprocessing.sequence.pad_sequences(self.Xtrain_sequence,
                                                                     padding='post')

        self.idx_word = dict(tokenizer.index_word)
        self.num_words = len(self.idx_word) + 1



    def load_word_embeddings(self, path = '../disaster_nlp/data/non_tracked/glove.6B.100d.txt'):
        '''
        sepcify path to e.g. glove word embeddings
        :return:
        '''
        self.word_embeddings = dict(load_embeddings(path))

    def match_word_embeddings(self):
        '''
        :return:
        '''
        embeddings_words = list(self.word_embeddings.keys())
        wordvec_dim = self.word_embeddings[embeddings_words[0]].shape[0]
        embedding_matrix = np.zeros((self.num_words, wordvec_dim))
        for i, word in self.idx_word.items():
            # Look up the word embedding
            vector = self.word_embeddings.get(word, None)
            # Record in matrix
            if vector is not None:
                embedding_matrix[i, :] = vector
        self.embedding_matrix = embedding_matrix


    def setup_net(self, epochs = 5, batch_size = 128):
        '''

        :return:
        '''
        self.nlabels = len(self.ytrain[0])
        wordvec_dim = self.embedding_matrix.shape[1]
        model_lstm = keras.Sequential()
        # initialise Ebedding layer num_words = len(idx_word) + 1 to deal with 0 padding
        # input_length is the number of words ids per sample e.g 28
        # NOT the sample size of the training data
        # you do not need to supply that info
        model_lstm.add(keras.layers.Embedding(input_dim=self.num_words,
                                              input_length=self.Xtrain_sequence.shape[1],
                                              output_dim=wordvec_dim,
                                              weights=[self.embedding_matrix],
                                              trainable=False,
                                              mask_zero=True))

        # words which are not in the pretrained embeddings (with value 0) are ignored
        model_lstm.add(keras.layers.Masking(mask_value=0.0))

        # Recurrent layer
        model_lstm.add(keras.layers.LSTM(200, return_sequences=False))
        model_lstm.add(keras.layers.Dropout(0.4))
        # model_lstm.add(keras.layers.LSTM(28, return_sequences=True))
        # model_lstm.add(keras.layers.Dropout(0.2))
        # model_lstm.add(keras.layers.LSTM(28, return_sequences=True))
        # model_lstm.add(keras.layers.Dropout(0.2))
        # model_lstm.add(keras.layers.LSTM(28, return_sequences=False))

        # Output layer
        model_lstm.add(keras.layers.Dense(self.nlabels))
        model_lstm.add(keras.layers.Activation('softmax'))

        # Compile the model
        model_lstm.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # model summary
        model_lstm.summary()

        ##fit
        model_lstm.fit(self.Xtrain_sequence, self.ytrain, epochs=epochs, batch_size=batch_size)
        self.model_lstm = model_lstm


refit = False
picklefile = 'LSTM_model.pickle'
#prepare data
if refit is True:
    dftrain = pd.read_csv('./data/datasets_117486_281522_atis.train.csv')
    Xtrain, ytrain = list(dftrain.values[:,1]), list(dftrain.values[:,-1])
    Xtrain = stop_word_removal(Xtrain,words=['is', 'the', 'of', 'a'])
    Xtrain = [Xt.replace('BOS ','').replace(' EOS','') for Xt in Xtrain]
    ytrain = pd.get_dummies(ytrain)
    labels = list(ytrain.columns)
    nlabels = len(labels)
    ytrain = np.array(ytrain)
    dftest = pd.read_csv('./data/datasets_117486_281522_atis.test.csv')
    Xtest, ytest = list(dftest.values[:,1]), list(dftest.values[:,-1])
    Xtest = [Xt.replace('BOS ','').replace(' EOS','') for Xt in Xtest]
    ytest = pd.get_dummies(ytest)
    ytest = np.array(ytest)


    #load the lstm model and perform tokenisation and load word embeddings
    x = lstm_fit()
    x.Xtrain = Xtrain
    x.Xtest = Xtest
    x.ytrain = ytrain
    x.ytest = ytest
    x.labels = labels

    x.tokenize()
    x.load_word_embeddings()
    x.match_word_embeddings()
    x.setup_net()

    #save to pickle
    picklefile = picklefile
    os.system('rm ' + picklefile)
    pickle_out = open(picklefile, "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
else:
    ## load previous simulation
    pickle_in = open(picklefile, "rb")
    x = pickle.load(pickle_in)

model_lstm = x.model_lstm

Xtest_tok =  x.tokenizer.texts_to_sequences(x.Xtest)
maxlen = x.Xtrain_sequence.shape[1]
Xtest_seq =  keras.preprocessing.sequence.pad_sequences(Xtest_tok,
                                                        padding='post',
                                                        maxlen=maxlen)
ypred_proba = model_lstm.predict(Xtest_seq)
ypred_class = [int(np.argmax(yp)) for yp in ypred_proba]
ypred_label = [x.labels[yp] for yp in ypred_class]