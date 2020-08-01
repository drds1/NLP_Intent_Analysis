import pandas as pd
import keras

dftrain = pd.read_csv('./data/datasets_117486_281522_atis.train.csv')
Xtrain, ytrain = list(dftrain.values[:,1]), list(dftrain.values[:,-1])
ytrain = pd.get_dummies(ytrain)
dftest = pd.read_csv('./data/datasets_117486_281522_atis.test.csv')
Xtest, ytest = list(dftest.values[:,1]), list(dftest.values[:,-1])
ytest = pd.get_dummies(ytest)

#convert all abstracts to sequences of integers key stored in idx_word
tokenizer = keras.preprocessing.text.Tokenizer(num_words=50,
                                               filters='?!":;,.#$&()*+-<=>@[\\]^_`{|}~\t\n',
                                               lower=True, split=' ')
tokenizer.fit_on_texts(Xtrain)
#assign number to each word
Xtrain_sequence = tokenizer.texts_to_sequences(Xtrain)
#padd the sequences of short sentences with 0s so everything is the same length
Xtrain_sequence = keras.preprocessing.sequence.pad_sequences(Xtrain_sequence)
idx_word = tokenizer.index_word


#model_lstm = keras.Sequential()
#model_lstm.add(keras.layers.Embedding(vocab_in_size, embedding_dim, input_length=len_input_train))
#model_lstm.add(keras.layers.LSTM(units))
#
##output the probability of the input belonging to each class
#model_lstm.add(keras.layers.Dense(nb_labels, activation='softmax'))
#model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_lstm.summary()
#
#history_lstm = model_lstm.fit(input_data_train, intent_data_label_cat_train,
#                              epochs=10,batch_size=BATCH_SIZE)