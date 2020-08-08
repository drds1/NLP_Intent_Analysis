from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
		'Good Good work',
		'Great effort',
		'nice work',
		'Excellent!',
        'really',
        'really',
        'really Good',
        'really work',
        'work under the really sea']
# create the tokenizer
t = Tokenizer(num_words=4)
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
sdocs = t.texts_to_sequences(docs)
print(encoded_docs)
print(sdocs)