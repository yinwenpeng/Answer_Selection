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
            line_control+=1
            #if line_control==100:
            #    break
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

def load_word2vec_to_init(rand_values):

    readFile=open('/mounts/data/proj/wenpeng/Dataset/insuranceQA/vocab_embs.txt', 'r')
    line_count=1
    for line in readFile:
        tokens=line.strip().split()
        rand_values[line_count]=numpy.array(map(float, tokens[1:]))
        line_count+=1                                            
    readFile.close()
    print 'initialization over...'
    return rand_values
    
