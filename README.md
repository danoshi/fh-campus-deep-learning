## fh-campus-deep-learning-chatbot
# Packages
This project consist of the packages below.
* numpy
* nltk
* tensorflow
* tflearn 
* json
* random

# Training Data 
The training data is from the following project https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/intents.json
Moreover, some cases are added by ourselve to extend the data.
This training data is a json file with specifc tags and patterns how this tag would appear in the conversation.
Afterwards we have some possiblities what to answer.
This data is used to train our neural network, it will take the users input and try to classify it to one of the tags.

# Data mofification
Every pattern will be in a list using the word_tokenizer class from nltk.
Another thing which has to be done is to "stemming a word" which means to find the root of the word.
Last but not least we need to give our neural network a numeric input, we will represent each sentence with a list of words.
This will give us the possiblities to return \0 if the word is not present and \1 if it is present.

# The model
Our model have two hidden layers, and the main goal of the network will be to map the words to the give tag in the intents file.
The epochs number is the amount of times that the model wee our data data during the training period.

