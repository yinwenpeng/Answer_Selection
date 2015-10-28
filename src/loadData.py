import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from operator import itemgetter

def load_ibm_corpus(vocabFile, trainFile, devFile, maxlength):
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    sentlength_limit=1040
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # label, question, answer
            #question
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().split()  
                length=len(words)
                if length>sentlength_limit:
                    words=words[:sentlength_limit]
                    length=sentlength_limit
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent+=[0]*left
                for word in words:
                    sent.append(word2id.get(word))
                sent+=[0]*right
                data.append(sent)
                del sent
                del words
            line_control+=1
            if line_control%100==0:
                print line_control
        read_file.close()
        return numpy.array(data),numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_dev_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().split() 
                length=len(words)
                if length>sentlength_limit:
                    words=words[:sentlength_limit]
                    length=sentlength_limit
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
                    sent.append(word2id.get(word))
                sent+=[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        return numpy.array(data),Y, numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_file(devFile, vocab)
    print 'dev file loaded over, total pairs: ', len(devLengths)/2
   

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y


    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    dev_left_pad=shared_dataset(devLeftPad)
    dev_right_pad=shared_dataset(devRightPad)
                                
    #valid_set_y = shared_dataset(devY)
    

    rval = [(indices_train,train_set_Lengths, train_left_pad, train_right_pad), (indices_dev, devY, valid_set_Lengths, dev_left_pad, dev_right_pad)]
    return rval, word_ind-1

def load_word2vec_to_init(rand_values, file):

    readFile=open(file, 'r')
    line_count=1
    for line in readFile:
        tokens=line.strip().split()
        rand_values[line_count]=numpy.array(map(float, tokens[1:]))
        line_count+=1                                            
    readFile.close()
    print 'initialization over...'
    return rand_values
    
def load_msr_corpus(vocabFile, trainFile, testFile, maxlength): #maxSentLength=60
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # label, sent1, sent2
            Y.append(int(tokens[0])) #repeat
            Y.append(int(tokens[0])) 
            #question
            for i in [1,2,2,1]: #shuffle the example
                sent=[]
                words=tokens[i].strip().lower().split()  
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1

                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
        read_file.close()
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), normalized_lengths, numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            #Y.append(int(tokens[0]))
            for i in [1,2]:
                sent=[]
                words=tokens[i].strip().lower().split()  
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1

                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), normalized_lengths, numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths,normalized_trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, normalized_testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
   

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1

def load_mts(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
        train_values.append(tokens)#repeat once
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values