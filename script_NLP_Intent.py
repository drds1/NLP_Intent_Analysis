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

#convert all abstracts to sequences of integers key stored in idx_word
tokenizer = keras.preprocessing.text.Tokenizer(num_words=100,
                                               filters="?!':;,.#$&()*+-<=>@[\\]^_`{|}~\t\n",
                                               lower=True, split=' ')


tokenizer.fit_on_texts(Xtrain)
word_counts = pd.DataFrame(dict(tokenizer.word_counts),index=['word count']).transpose()
word_counts.sort_values(by='word count',ascending = False,inplace=True)
print(word_counts.head(10))
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

#just test embedding layer
#model_lstm.compile('rmsprop', 'mse')
#embedding_test = model_lstm.predict(Xtrain_sequence)
#Xtrain_sequence.shape ... (4274, 28)
#embedding_test.shape ... (4274, 28. 100) i.e. each word converted to 100d vector


#words which are not in the pretrained embeddings (with value 0) are ignored
load_model = False
picklefile = 'LSTM_model.pickle'

model_lstm.add(keras.layers.Masking(mask_value = 0.0))

# Recurrent layer
model_lstm.add(keras.layers.LSTM(200, return_sequences=False))
model_lstm.add(keras.layers.Dropout(0.4))
#model_lstm.add(keras.layers.LSTM(28, return_sequences=True))
#model_lstm.add(keras.layers.Dropout(0.2))
#model_lstm.add(keras.layers.LSTM(28, return_sequences=True))
#model_lstm.add(keras.layers.Dropout(0.2))
#model_lstm.add(keras.layers.LSTM(28, return_sequences=False))


# Dropout for regularisation and avoid overfit
#model_lstm.add(keras.layers.Dropout(0.2))

# Output layer
model_lstm.add(keras.layers.Dense(nlabels))
model_lstm.add(keras.layers.Activation('softmax'))

# Compile the model
model_lstm.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model summary
model_lstm.summary()

##fit
model_lstm.fit(Xtrain_sequence, ytrain,epochs=5,batch_size=128)

#save fitted model
picklefile = picklefile
os.system('rm ' + picklefile)
pickle_out = open(picklefile, "wb")
pickle.dump(model_lstm, pickle_out)
pickle_out.close()

## load previous simulation
#pickle_in = open(picklefile, "rb")
#model_lstm = pickle.load(pickle_in)

##predict values for testing
Xtest_sequence = tokenizer.texts_to_sequences(Xtest)
Xtest_sequence = keras.preprocessing.sequence.pad_sequences(Xtest_sequence,
                                                            maxlen = Xtrain_sequence.shape[1],
                                                            padding='post')


eval_lstm = model_lstm.evaluate(Xtest_sequence, ytest)
ypred = model_lstm.predict(Xtest_sequence)



X0 = Xtrain_sequence[0,:]
x00 = ''
for x in X0:
    x00+= idx_word.get(x,'')+' '
print(x00)
print(Xtrain[0])
'''

        input1 = Input(shape=(max_words,))
        embedding_layer1 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words,
                                     trainable=False)(input1)
        dropout1 = Dropout(0.2)(embedding_layer1)
        lstm1_1 = LSTM(128, return_sequences=True)(dropout1)
        lstm1_2 = LSTM(128, return_sequences=True)(lstm1_1)
        lstm1_2a = LSTM(128, return_sequences=True)(lstm1_2)
        lstm1_3 = LSTM(128)(lstm1_2a)

        input2 = Input(shape=(max_words_ky,))
        embedding_layer2 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words_ky,
                                     trainable=False)(
            input2)
        dropout2 = Dropout(0.2)(embedding_layer2)
        lstm2_1 = LSTM(64, return_sequences=True)(dropout2)
        lstm2_2 = LSTM(64, return_sequences=True)(lstm2_1)
        lstm2_3 = LSTM(64)(lstm2_2)

        input3 = Input(shape=(max_words_lc,))
        embedding_layer3 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words_lc,
                                     trainable=False)(
            input3)
        dropout3 = Dropout(0.2)(embedding_layer3)
        lstm3_1 = LSTM(32, return_sequences=True)(dropout3)
        lstm3_2 = LSTM(32, return_sequences=True)(lstm3_1)
        lstm3_3 = LSTM(32)(lstm3_2)

        merge = concatenate([lstm1_3, lstm2_3, lstm3_3])

        dropout = Dropout(0.8)(merge)
        dense1 = Dense(256, activation='relu')(dropout)
        dense2 = Dense(128, activation='relu')(dense1)
        output = Dense(2, activation='softmax')(dense2)
        model = Model(inputs=[input1, input2, input3], outputs=output)
        model.summary()
'''