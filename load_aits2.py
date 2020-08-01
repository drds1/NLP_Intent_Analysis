import pandas as pd
import keras

dftrain = pd.read_csv('./data/datasets_117486_281522_atis.train.csv')
Xtrain, ytrain = list(dftrain.values[:,1]), list(dftrain.values[:,-1])
ytrain = pd.get_dummies(ytrain)
dftest = pd.read_csv('./data/datasets_117486_281522_atis.test.csv')
Xtest, ytest = list(dftest.values[:,1]), list(dftest.values[:,-1])
ytest = pd.get_dummies(ytest)

tokenizer = keras.preprocessing.text.Tokenizer(num_words=50,
                                               filters='?!":;,.#$&()*+-<=>@[\\]^_`{|}~\t\n',
                                               lower=True, split=' ')

#tokenise the input
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer()
Xtrain_counts = count_vect.fit_transform(Xtrain)
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(Xtrain_counts)
Xtrain_tfidf = tfidf_transformer.transform(Xtrain_counts)


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