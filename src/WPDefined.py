import cPickle
import gzip
import os
import sys
import time

import numpy

import theano.sandbox.neighbours as TSN
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from cis.deep.utils.theano import debug_print

def load_model_for_training(wikiFile, trainFile, devFile, maxlength, dataMode, train_scheme):

    def store_word2id_into_file(word2id):
        if train_scheme ==1:
            output = open('/mounts/data/proj/wenpeng/CNN_LM/word2id_onlytrain.txt', 'w')
        elif train_scheme ==2:
            output = open('/mounts/data/proj/wenpeng/CNN_LM/word2id_onlywiki.txt', 'w')
        for word in word2id:
            output.write(word+"\t"+str(word2id[word])+"\n")
        print 'word2id stored over.'
        output.close()

    def load_train_trainOfSenti(file):   
        id2count={}
        word2id={}
        senti_file=open(file, 'r')
        data=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        word_count=1
        for line in senti_file:
            #pre_index=0
            tokens=line.strip().split('\t')
            #Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            length=len(words)
            #total=total+length
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            sent+=[0]*left
            for word in words:
                #sent.append(word2id.get(word))
                    
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:
                    #embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                    #embeddings_target.append(numpy.random.uniform(-1,1,embedding_size))
                    word2id[word]=word_count   # starts from 1
                    id2count[word_count]=1 #1 means new words
                    sent.append(word_count)
                    word_count=word_count+1                  
                else:
                    sent.append(id)
                    id2count[id]=id2count[id]+1# add 1 for kown words
            sent+=[0]*right
            data.append(sent)
        senti_file.close()
        print '\nVocab:'+str(len(word2id))
        #compute word occurring probability
        all_tokens=sum(Lengths)
        for key in id2count:
            id2count[key]=id2count[key]*1.0/all_tokens  #discard the first zero word
        unigram=[]
        word_count=word_count-1  #word_count is 1 more than the actual new words
        for index in xrange(word_count):
            unigram.append(id2count[index+1])            
        store_word2id_into_file(word2id)
        return numpy.array(data),numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad), word_count, word2id, numpy.array(unigram)
    
    def load_train_wikipedia(file):   
        id2count={}
        word2id={}
        senti_file=open(file, 'r')
        data=[]
        total_lines=74915588 
        Lengths=[]
        leftPad=[]
        rightPad=[]
        j=0
        word_count=1 # word index start from 1
        for line in senti_file:
            if len(line.strip())>0: #not empty lines
                
                #if j==2000:
                #    break
                words=line.strip().lower().split(' ') #convert to lowercase
                sent=[]
                length=len(words)
                
                if length > maxlength or length < 3:
                    continue
                #a nice sentence
                j=j+1
                sys.stdout.write( "Loading wikipedia :[%6f] %% complete!\r" % (j*100.0/total_lines) )
                sys.stdout.flush()
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)   
                #for i in range(left):
                #    sent.append(0)
                sent+=[0]*left
                for word in words:
                    #sent.append(word2id.get(word))
                    
                    id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                    if id == -1:
                        #embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                        #embeddings_target.append(numpy.random.uniform(-1,1,embedding_size))
                        word2id[word]=word_count   # starts from 1
                        id2count[word_count]=1 #1 means new words
                        sent.append(word_count)
                        word_count=word_count+1                  
                    else:
                        sent.append(id)
                        id2count[id]=id2count[id]+1# add 1 for kown words
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print '\nVocab:'+str(len(word2id))
        #compute word occurring probability
        all_tokens=sum(Lengths)
        for key in id2count:
            id2count[key]=id2count[key]*1.0/all_tokens  #discard the first zero word
        unigram=[]
        word_count=word_count-1  #word_count is 1 more than the actual new words
        for index in xrange(word_count):
            unigram.append(id2count[index+1])
            
        store_word2id_into_file(word2id)
        return numpy.array(data),numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad), word_count, word2id, numpy.array(unigram)
    def load_dev_or_test_senti(file, word2id):
        senti_file=open(file, 'r')
        data=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        #total=0
        for line in senti_file:
            #pre_index=0
            tokens=line.strip().split('\t')
            #Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            length=len(words)
            #total=total+length
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            #for i in range(left):
            #    sent.append(0)
            sent+=[0]*left
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:    
                    unknown=unknown+1              
                    #sent.append(numpy.random.random_integers(word_count)) 
                    #sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero      
                    sent.append(0)           
                else:
                    sent.append(id)
                    #pre_index=id
            #for i in range(right):
            #    sent.append(0)
            sent+=[0]*right
            data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/sum(Lengths))
        return numpy.array(data), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_dev_or_test_senti_preIndex(file, word2id):
        senti_file=open(file, 'r')
        data=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        #total=0
        for line in senti_file:
            pre_index=0
            tokens=line.strip().split('\t')
            #Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            length=len(words)
            #total=total+length
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            sent+=[0]*left
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:       
                    unknown=unknown+1           
                    #sent.append(numpy.random.random_integers(word_count)) 
                    sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero                
                else:
                    sent.append(id)
                    pre_index=id
            sent+=[0]*right
            data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/sum(Lengths))
        return numpy.array(data), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_dev_or_test_senti_skipUnknown(file, word2id):
        senti_file=open(file, 'r')
        data=[]
        #Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            tokens=line.strip().split('\t')
            #Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            total=total+len(words)
            knownWords=[]
            for word in words:
                id=word2id.get(word, -1) 
                if id != -1:
                    knownWords.append(word)
                else:
                    #print 'unknown word: '+word
                    unknown=unknown+1
            
            length=len(knownWords)
            
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            sent+=[0]*left
            for word in knownWords:
                sent.append(word2id.get(word, -1))
            sent+=[0]*right
            data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    
    if train_scheme == 1:
        indices_train, trainLengths, trainLeftPad, trainRightPad, word_count, word2id, unigram=load_train_trainOfSenti(trainFile)
    elif train_scheme == 2:
        indices_train, trainLengths, trainLeftPad, trainRightPad, word_count, word2id, unigram=load_train_wikipedia(wikiFile)
    print 'train file loaded over, sentence totally:'+str(len(trainLengths))
    if dataMode==1:
        indices_dev, devLengths, devLeftPad, devRightPad=load_dev_or_test_senti(devFile, word2id)
        print 'dev file loaded over, sentence totally:'+str(len(devLengths))
    elif dataMode==2:
        indices_dev, devLengths, devLeftPad, devRightPad=load_dev_or_test_senti_preIndex(devFile, word2id)
        print 'dev file loaded over, sentence totally:'+str(len(devLengths))
    elif dataMode==3:
        indices_dev, devLengths, devLeftPad, devRightPad=load_dev_or_test_senti_skipUnknown(devFile, word2id)
        print 'dev file loaded over, sentence totally:'+str(len(devLengths))    

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y

    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    #uni_gram=shared_dataset(unigram)
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    dev_left_pad=shared_dataset(devLeftPad)
    dev_right_pad=shared_dataset(devRightPad)
    

    rval = [(indices_train,train_set_Lengths, train_left_pad, train_right_pad), (indices_dev, valid_set_Lengths, dev_left_pad, dev_right_pad)]
    return rval, unigram, trainLengths, devLengths,word_count

def read_data_WP(trainFile, devFile, testFile, emb_file, maxlength, useEmb, dataMode):
    #first store emb_file into a dict
    embeddings=[]
    embeddings_Q=[]
    word2id={}
    embedding_size=0
    if useEmb:
        embeddingsFile=open(emb_file,'r')
    
        for num_lines, line in enumerate(embeddingsFile):
            
            #if num_lines > 99:
            #    break
            
            tokens=line.strip().split() # split() with no parameters means seperating using consecutive spaces
            vector=[]
            embedding_size=len(tokens)-1
            for i in range(1, embedding_size+1):
                vector.append(float(tokens[i]))
            if num_lines==0:
                embeddings.append(numpy.zeros(embedding_size)) 
                embeddings_Q.append(numpy.zeros(embedding_size)) 
            embeddings.append(vector)
            embeddings_Q.append(vector)
            word2id[tokens[0]]=num_lines+1 # word index starts from 1
    
        embeddingsFile.close()
    else:
        embedding_size=48   #the paper uses 48
        embeddings.append(numpy.zeros(embedding_size)) 
        embeddings_Q.append(numpy.zeros(embedding_size)) 
    word_count=len(embeddings)
    print 'Totally, '+str(word_count)+' word embeddings.'
    
    def load_train_file(file, embeddings, embeddings_target, word_count, word2id):   
        id2count={}
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        for line in senti_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            length=len(words)
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)   
            for i in range(left):
                sent.append(0)
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:
                    embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                    embeddings_target.append(numpy.random.uniform(-1,1,embedding_size))
                    word2id[word]=word_count
                    id2count[word_count]=1 #1 means new words
                    sent.append(word_count)
                    word_count=word_count+1                  
                else:
                    sent.append(id)
                    id2count[id]=id2count[id]+1# add 1 for kown words
            for i in range(right):
                sent.append(0)
            data.append(sent)
        senti_file.close()
        print 'Vocab:'+str(len(word2id))
        #compute word occurring probability
        all_tokens=sum(Lengths)
        for key in id2count:
            id2count[key]=id2count[key]*1.0/all_tokens  #discard the first zero word
        unigram=[]
        for index in xrange(word_count-1):
            unigram.append(id2count[index+1])
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad), numpy.array(embeddings), numpy.array(embeddings_target), word_count, word2id, numpy.array(unigram)
    def load_dev_or_test_file(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            #pre_index=0
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            length=len(words)
            total=total+length
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            for i in range(left):
                sent.append(0)
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:    
                    unknown=unknown+1              
                    #sent.append(numpy.random.random_integers(word_count)) 
                    #sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero      
                    sent.append(0)           
                else:
                    sent.append(id)
                    #pre_index=id
            for i in range(right):
                sent.append(0)
            data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_dev_or_test_file_preIndex(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            pre_index=0
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            length=len(words)
            total=total+length
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            for i in range(left):
                sent.append(0)
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:       
                    unknown=unknown+1           
                    #sent.append(numpy.random.random_integers(word_count)) 
                    sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero                
                else:
                    sent.append(id)
                    pre_index=id
            for i in range(right):
                sent.append(0)
            data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_dev_or_test_file_skipUnknown(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].lower().split(' ')    # all words converted into lowercase
            knownWords=[]
            for word in words:
                id=word2id.get(word, -1) 
                if id != -1:
                    knownWords.append(word)
                else:
                    unknown=unknown+1
            
            length=len(knownWords)
            total=total+length
            Lengths.append(length)
            left=(maxlength-length)/2
            right=maxlength-left-length
            leftPad.append(left)
            rightPad.append(right)
            if left<0 or right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            for i in range(left):
                sent.append(0)
            for word in knownWords:
                sent.append(word2id.get(word, -1))
            for i in range(right):
                sent.append(0)
            data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad, embeddings, embeddings_Q, word_count, word2id, unigram=load_train_file(trainFile, embeddings, embeddings_Q, word_count, word2id)
    print 'train file loaded over, totally:'+str(len(trainLengths))
    if dataMode==1:
        indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_or_test_file(devFile, word_count, word2id)
        print 'dev file loaded over, totally:'+str(len(devLengths))
        indices_test, testY, testLengths, testLeftPad, testRightPad=load_dev_or_test_file(testFile, word_count, word2id)
        print 'test file loaded over, totally:'+str(len(testLengths))
    elif dataMode==2:
        indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_or_test_file_preIndex(devFile, word_count, word2id)
        print 'dev file loaded over, totally:'+str(len(devLengths))
        indices_test, testY, testLengths, testLeftPad, testRightPad=load_dev_or_test_file_preIndex(testFile, word_count, word2id)   
        print 'test file loaded over, totally:'+str(len(testLengths))
    elif dataMode==3:
        indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_or_test_file_skipUnknown(devFile, word_count, word2id)
        print 'dev file loaded over, totally:'+str(len(devLengths))
        indices_test, testY, testLengths, testLeftPad, testRightPad=load_dev_or_test_file_skipUnknown(testFile, word_count, word2id)   
        print 'test file loaded over, totally:'+str(len(testLengths))     

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y

    embeddings_theano = theano.shared(numpy.asarray(embeddings, dtype=theano.config.floatX), borrow=True)  # @UndefinedVariable
    embeddings_Q_theano= theano.shared(numpy.asarray(embeddings_Q, dtype=theano.config.floatX), borrow=True)
    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    test_set_Lengths = shared_dataset(testLengths)
    #uni_gram=shared_dataset(unigram)
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    dev_left_pad=shared_dataset(devLeftPad)
    dev_right_pad=shared_dataset(devRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
    
    train_set_y=shared_dataset(trainY)                             
    valid_set_y = shared_dataset(devY)
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y,train_set_Lengths, train_left_pad, train_right_pad), (indices_dev, valid_set_y, valid_set_Lengths, dev_left_pad, dev_right_pad), (indices_test, test_set_y, test_set_Lengths, test_left_pad, test_right_pad)]
    return rval,      embedding_size, embeddings_theano, embeddings_Q_theano, unigram, trainLengths, devLengths, testLengths

class ConvFoldPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=4, left=[], right=[]):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')
        
        #folding
        matrix_shape=T.cast(T.join(0,
                            T.as_tensor([T.prod(conv_out.shape[:-1])]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        matrix = T.reshape(conv_out, matrix_shape, ndim=2)
        odd_matrix=matrix[0:matrix_shape[0]:2]
        even_matrix=matrix[1:matrix_shape[0]:2]
        raw_folded_matrix=odd_matrix+even_matrix
        
        out_shape=T.cast(T.join(0,  conv_out.shape[:-2],
                            T.as_tensor([conv_out.shape[2]/2]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        fold_out=T.reshape(raw_folded_matrix, out_shape, ndim=4)
        #ktop-max pooling
        matrices=[]
        for i in range(image_shape[0]): # image_shape[0] is actually batch_size
            neighborsForPooling = TSN.images2neibs(ten4=fold_out[i:(i+1)], neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')
            non_zeros=neighborsForPooling[:,left[i]:(neighborsForPooling.shape[1]-right[i])]
            #neighborsForPooling=neighborsForPooling[:,leftBound:(rightBound+1)] # only consider non-zero elements
            
            neighborsArgSorted = T.argsort(non_zeros, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-k:]
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie

            ii = T.repeat(T.arange(non_zeros.shape[0]), k)
            jj = kNeighborsArgSorted.flatten()
            pooledkmaxTmp = non_zeros[ii, jj] # now, should be a vector
            '''
            new_shape = T.cast(T.join(0, 
                           T.as_tensor([non_zeros.shape[0]]),
                           T.as_tensor([k])),
                           'int64')
            pooledkmaxTmp = T.reshape(pooledkmaxTmp, new_shape, ndim=2)
            '''
            matrices.append(pooledkmaxTmp)
        
        overall_matrix=T.concatenate(matrices, axis=0)         
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([k])),
                           'int64')
        pooled_out = T.reshape(overall_matrix, new_shape, ndim=4)      
        
    
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
        
def conv_WP(inputs, filters_W, filter_shape, image_shape):
    new_filter_shape=(filter_shape[0], filter_shape[1], 1, filter_shape[3])
    conv_outs=[]

    for i in range(image_shape[2]):#each dimension
        conv_out_i=conv.conv2d(input=inputs, filters=filters_W[:,:,i:(i+1),:],filter_shape=new_filter_shape, image_shape=image_shape, border_mode='full')
        conv_outs.append(conv_out_i[:,:,i:(i+1),:])
    overall_conv_out=T.concatenate(conv_outs, axis=2)
    
    return overall_conv_out
class Conv_Fold_DynamicK_PoolLayer_NAACL(object):
    """Pool Layer of a convolutional network """
    
        
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=8, unifiedWidth=30, left=3, right=3, W=2, b=2, firstLayer=True):
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # the original one
        
        self.W = W
        self.b = b

        # the bias is a 1D tensor -- one bias per output feature map
        #b_values = numpy.zeros((image_shape[2]/2,1), dtype=theano.config.floatX)
        #self.b = theano.shared(value=b_values, borrow=True)
        #bb=T.repeat(self.b, unifiedWidth, axis=1)
        conv_out=  conv_WP(inputs=input, filters_W=self.W, filter_shape=filter_shape, image_shape=image_shape)
        # convolve input feature maps with filters
        #conv_out = conv.conv2d(input=input, filters=self.W,
        #        filter_shape=filter_shape, image_shape=image_shape, border_mode='full')
        #folding
        matrix_shape=T.cast(T.join(0,
                            T.as_tensor([T.prod(conv_out.shape[:-1])]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        matrix = T.reshape(conv_out, matrix_shape, ndim=2)
        odd_matrix=matrix[0:matrix_shape[0]:2]
        even_matrix=matrix[1:matrix_shape[0]:2]
        raw_folded_matrix=(odd_matrix+even_matrix)*0.5
        '''
        out_shape=T.cast(T.join(0,  conv_out.shape[:-2],
                            T.as_tensor([conv_out.shape[2]/2]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        fold_out=T.reshape(raw_folded_matrix, out_shape, ndim=4)
        '''
        
        #padded_matrices=[]
        #for i in range(image_shape[0]): # image_shape[0] is actually batch_size
        #neighborsForPooling = TSN.images2neibs(ten4=fold_out[i:(i+1)], neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')
        #wenpeng1=theano.printing.Print('original')(neighborsForPooling[:, 25:35])
        self.fold_output=raw_folded_matrix.reshape((self.input.shape[0], filter_shape[0], self.input.shape[2]/2, poolsize[1]))
        non_zeros=raw_folded_matrix[:,left:-right] # only consider non-zero elements
        
        #wenpeng2=theano.printing.Print('non-zeros')(non_zeros)

        neighborsArgSorted = T.argsort(non_zeros, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-k:]
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie

        ii = T.repeat(T.arange(non_zeros.shape[0]), k)
        jj = kNeighborsArgSorted.flatten()
        pooledkmaxList = non_zeros[ii, jj] # now, should be a vector
        new_shape = T.cast(T.join(0, T.as_tensor([non_zeros.shape[0]]),T.as_tensor([k])),'int64')
        pooledkmaxMatrix = T.reshape(pooledkmaxList, new_shape, ndim=2)
        if firstLayer:
            leftWidth=(unifiedWidth-k)/2
            rightWidth=unifiedWidth-leftWidth-k
                
            left_padding = T.zeros((non_zeros.shape[0], leftWidth), dtype=theano.config.floatX)
            right_padding = T.zeros((non_zeros.shape[0], rightWidth), dtype=theano.config.floatX)
            pooledkmaxMatrix = T.concatenate([left_padding, pooledkmaxMatrix, right_padding], axis=1) 
            #padded_matrices.append(matrix_padded)     
        #else:
            #padded_matrices.append(pooledkmaxMatrix)
        '''                    
        overall_matrix=T.concatenate(padded_matrices, axis=0)         
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([unifiedWidth])),
                           'int64')
        pooled_out = T.reshape(overall_matrix, new_shape, ndim=4)
        '''
        pooled_out=pooledkmaxMatrix.reshape((self.input.shape[0],filter_shape[0], self.input.shape[2]/2,unifiedWidth))
        #wenpeng2=theano.printing.Print('pooled_out')(pooled_out[:,:,:,15:])
        # downsample each feature map individually, using maxpooling
        '''
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        '''
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #@wenpeng:  following tanh operation will voilate our expectation that zero-padding, for its output will have no zero any more
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #biased_pooled_out=pooled_out + bb.dimshuffle('x', 'x', 0, 1)

        #now, reset some zeros
        self.leftPad=(unifiedWidth-k)/2
        self.rightPad=unifiedWidth-self.leftPad-k
        '''
        if firstLayer:
            zero_recover_matrices=[]
            for i in range(image_shape[0]): # image_shape[0] is actually batch_size
                neighborsForPooling = TSN.images2neibs(ten4=biased_pooled_out[i:(i+1)], neib_shape=(1,biased_pooled_out.shape[3]), mode='ignore_borders')     
                left_zeros=T.set_subtensor(neighborsForPooling[:,:self.leftPad[i]], T.zeros((neighborsForPooling.shape[0], self.leftPad[i]), dtype=theano.config.floatX))
                right_zeros=T.set_subtensor(left_zeros[:,(neighborsForPooling.shape[1]-self.rightPad[i]):], T.zeros((neighborsForPooling.shape[0], self.rightPad[i]), dtype=theano.config.floatX))   
                zero_recover_matrices.append(right_zeros)
            overall_matrix_new=T.concatenate(zero_recover_matrices, axis=0)  
            pooled_out_with_zeros = T.reshape(overall_matrix_new, new_shape, ndim=4) 
            self.output=T.tanh(pooled_out_with_zeros)
        else:
            self.output=T.tanh(biased_pooled_out)
        '''
        self.output=T.tanh(pooled_out)

        # store parameters of this layer
        self.params = [self.W]



class HS_convolution_simplified(object):
    """Pool Layer of a convolutional network """
    '''
    this is a simplified version of original CNN_LM_HS: only one layers of wide-one-convolution, no folder, and finally max-pooling is used to form a embedding
    for the sentence, and added with context words to predict the middle word. The only change is to remove folding
    '''    
        
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=[], unifiedWidth=30, left=[], right=[], firstLayer=True):
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        # the original one
        
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)
        '''
        self.W = theano.shared(value=numpy.zeros(filter_shape,
                                                 dtype=theano.config.floatX),  # @UndefinedVariable
                                name='W', borrow=True)
        '''
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        #bb=T.repeat(self.b, unifiedWidth, axis=1)
        conv_out= debug_print( conv_WP(inputs=input, filters_W=self.W, filter_shape=filter_shape, image_shape=image_shape), 'conv_out')
        # convolve input feature maps with filters
        #conv_out = conv.conv2d(input=input, filters=self.W,
        #        filter_shape=filter_shape, image_shape=image_shape, border_mode='full')
        #folding
        fold_out=conv_out
        
        
        padded_matrices=[]
        for i in range(image_shape[0]): # image_shape[0] is actually batch_size
            neighborsForPooling = TSN.images2neibs(ten4=fold_out[i:(i+1)], neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')
            #wenpeng1=theano.printing.Print('original')(neighborsForPooling[:, 25:35])

            non_zeros=neighborsForPooling[:,left[i]:(neighborsForPooling.shape[1]-right[i])] # only consider non-zero elements
            #wenpeng2=theano.printing.Print('non-zeros')(non_zeros)

            neighborsArgSorted = T.argsort(non_zeros, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-k[i]:]
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie

            ii = T.repeat(T.arange(non_zeros.shape[0]), k[i])
            jj = kNeighborsArgSorted.flatten()
            pooledkmaxList = non_zeros[ii, jj] # now, should be a vector
            new_shape = T.cast(T.join(0, 
                           T.as_tensor([non_zeros.shape[0]]),
                           T.as_tensor([k[i]])),
                           'int64')
            pooledkmaxMatrix = debug_print(T.reshape(pooledkmaxList, new_shape, ndim=2), 'pooledkmaxMatrix')
            #in simplified version, following is always top layer, so no padding
            if firstLayer:
                leftWidth=(unifiedWidth-k[i])/2
                rightWidth=unifiedWidth-leftWidth-k[i]
                
                left_padding = T.zeros((non_zeros.shape[0], leftWidth), dtype=theano.config.floatX)
                right_padding = T.zeros((non_zeros.shape[0], rightWidth), dtype=theano.config.floatX)
                matrix_padded = T.concatenate([left_padding, pooledkmaxMatrix, right_padding], axis=1) 
                padded_matrices.append(matrix_padded)     
            else:
                padded_matrices.append(pooledkmaxMatrix)
                            
        overall_matrix=debug_print(T.concatenate(padded_matrices, axis=0), 'overall_matrix')
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([1])),
                           'int64')
        pooled_out = debug_print( T.reshape(overall_matrix, new_shape, ndim=4), 'pooled_out')
        #wenpeng2=theano.printing.Print('pooled_out')(pooled_out[:,:,:,15:])
        # downsample each feature map individually, using maxpooling
        '''
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        '''
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


        # store parameters of this layer
        self.params = [self.W, self.b]


       
def dropout_from_layer(rng,layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output



def shared_dataset(data_y, borrow=True):
    shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
    return T.cast(shared_y, 'int32')

class Conv_KmaxPool_Layer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=4):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        #images2neibs produces a 2D matrix
        neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')

        #k = poolsize[1]

        neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-k:]
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)

        ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
        jj = kNeighborsArgSorted.flatten()
        pooledkmaxTmp = neighborsForPooling[ii, jj]

        # reshape pooledkmaxTmp
        new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                           T.as_tensor([conv_out.shape[2]]),
                           T.as_tensor([k])),
                           'int64')
        pooled_out = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
        
        # downsample each feature map individually, using maxpooling
        '''
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        '''
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class FullyConnectedLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, rng, input, n_in, n_out, useTanh):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        '''
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),  # @UndefinedVariable
                                name='W', borrow=True)
        '''
        self.W= theano.shared(numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX), borrow=True)
        
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),  # @UndefinedVariable
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form, 
        if useTanh:
            self.output = T.tanh(T.dot(input, self.W) + self.b)  # is a vector
        else:
            self.output = T.dot(input, self.W) + self.b  # is a vector
        self.params = [self.W, self.b]

class SoftMaxlayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input):

        # compute vector of class-membership probabilities in symbolic form, 
        self.p_y_given_x = T.nnet.softmax(input)  # is a vector
        self.y_pred = T.argmax(self.p_y_given_x, axis=1) # choose the maximal element in above vector



    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    #wenpeng define cross-entropy for logistic_sgd
    def cross_entropy_regularization(self, y, params):
        
        cost =-T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        '''
        for params_i in params:
            cost=cost+0.01*(params_i**2).sum()
        '''
        return cost
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))   # wenpeng modified from "('y', target.type, 'y_pred', self.y_pred.type))"
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def repeat_whole_matrix(tensor, n, row_dir=True):
    repeated_raw=T.repeat(tensor, n, axis=3)
    list_tensor=[]
    for i in xrange(n):
        list_tensor.append(repeated_raw[:,:,:,i::n])
    if row_dir: #shape[2] will change
        return T.concatenate(list_tensor, axis=2)
    else:
        return T.concatenate(list_tensor, axis=3)

def repeat_whole_tensor(matrix, n, row_dir=True):
    repeated_raw=T.repeat(matrix, n, axis=1)
    list_matrix=[]
    for i in xrange(n):
        list_matrix.append(repeated_raw[:,i::n])
    if row_dir: #shape[2] will change
        return T.concatenate(list_matrix, axis=0)
    else:
        return T.concatenate(list_matrix, axis=1)
    
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if numpy.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break    