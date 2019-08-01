import nltk.data
import exceptions
import numpy as np
from nltk.tokenize import RegexpTokenizer

def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            #print word, model[word]
            return model[word].reshape(1, 300)
        else:
            #print ('Word missing w2v: ' + word)
            #return model['UNK'].reshape(1, 300)
            #return np.random.randn(1, 300)
            return np.zeros([1, 300])

def location_model(loc, i, offset, token_num):
    print "location", loc, i, offset, token_num
    loc_embed = np.zeros([1, 300])
    loc_embed[0] = (i+1)*1.0/(token_num+1)
    return loc_embed

