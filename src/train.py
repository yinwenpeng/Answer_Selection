
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
from loadData import load_ibm_corpus, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal


def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[50], batch_size=10, window_width=3,
                    maxSentLength=1050, emb_size=50, hidden_size=200,
                    margin=0.5):
#def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[6, 12], batch_size=70, useAllSamples=0, kmax=30, ktop=5, filter_size=[10,7],
#                    L2_weight=0.000005, dropout_p=0.5, useEmb=0, task=5, corpus=1):

    ibmPath='/mounts/data/proj/wenpeng/Dataset/insuranceQA/';


    rng = numpy.random.RandomState(23455)
    datasets, vocab_size=load_ibm_corpus(ibmPath+'vocabulary', ibmPath+'train.txt', ibmPath+'dev.txt', maxSentLength)

    indices_train, trainLengths, trainLeftPad, trainRightPad= datasets[0]
    #print trainY.eval().shape[0]
    indices_dev, devY, devLengths, devLeftPad, devRightPad= datasets[1]

    n_train_batches=indices_train.shape[0]/(batch_size*4) #note that we consider 4 lines as an example in training
    n_valid_batches=indices_dev.shape[0]/(batch_size*4)
    
    train_batch_start=list(numpy.arange(n_train_batches)*(batch_size*4))
    dev_batch_start=list(numpy.arange(n_valid_batches)*(batch_size*4))

    indices_train_theano=theano.shared(numpy.asarray(indices_train, dtype=theano.config.floatX), borrow=True)
    indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
    indices_train_theano=T.cast(indices_train_theano, 'int32')
    indices_dev_theano=T.cast(indices_dev_theano, 'int32')



    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(1e-50+numpy.zeros(emb_size))
    #rand_values[0]=numpy.array([1e-50]*emb_size)
    rand_values=load_word2vec_to_init(rand_values)
    embeddings=theano.shared(value=rand_values)      

    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x_index = T.imatrix('x_index')   # now, x is the index matrix, must be integer
    #left=T.ivector('left')
    #right=T.ivector('right')
    
    #x=embeddings[x_index.flatten()].reshape(((batch_size*4),maxSentLength, emb_size)).transpose(0, 2, 1).flatten()
    ishape = (emb_size, maxSentLength)  # this is the size of MNIST images
    filter_size=(emb_size,window_width)
    #poolsize1=(1, ishape[1]-filter_size[1]+1) #?????????????????????????????
    length_after_wideConv=ishape[1]+filter_size[1]-1
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #layer0_input = x.reshape(((batch_size*4), 1, ishape[0], ishape[1]))
    layer0_input = embeddings[x_index.flatten()].reshape(((batch_size*4),maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)

    #layer0_output = debug_print(layer0.output, 'layer0.output')
    layer0 = Conv(rng, input=layer0_input,
            image_shape=((batch_size*4), 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]))
    
    layer0_out=debug_print(layer0.output, 'layer0_out')
    
    layer1=Average_Pooling(rng, input=layer0_out, length_last_dim=length_after_wideConv, kern=nkerns[0] ) 
    
    layer1_out=debug_print(layer1.output.reshape((batch_size*2, nkerns[0]*2)), 'layer1_out')
    
    layer2=HiddenLayer(rng, input=layer1_out, n_in=nkerns[0]*2, n_out=hidden_size, activation=T.tanh)
    layer3=HiddenLayer(rng, input=layer2.output, n_in=hidden_size, n_out=1, activation=T.tanh)
    
    posi_score=layer3.output[0:layer3.output.shape[0]:2,:]
    nega_score=layer3.output[1:layer3.output.shape[0]:2,:]
    cost=T.maximum(0, margin-T.sum(posi_score-nega_score))
    

    
    #cost = layer3.negative_log_likelihood(y)
    # output a list of score
    dev_model = theano.function([index], layer3.output.flatten(),
             givens={
                x_index: indices_dev_theano[index: index + (batch_size*4)]})
    # create a list of all model parameters to be fit by gradient descent

    params = layer3.params + layer2.params + layer1.params+layer0.params
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-20)))   #AdaGrad
        updates.append((acc_i, acc))    
  
    train_model = theano.function([index], [cost], updates=updates,
          givens={
            x_index: indices_train_theano[index: index + (batch_size*4)]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches/5, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1

            minibatch_index=minibatch_index+1
            cost_ij= train_model(batch_start)
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+' cost: '+str(cost_ij)
            if iter % validation_frequency == 0:
                dev_scores=[]
                for i in dev_batch_start:
                    dev_scores+=list(dev_model(i))

                acc_dev=compute_acc(devY, dev_scores)
                print(('\t\t\t\tepoch %i, minibatch %i/%i, dev acc of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,
                           acc_dev * 100.))

                '''
                #print 'validating & testing...'
                # compute zero-one loss on validation set
                validation_losses = []
                for i in dev_batch_start:
                    time.sleep(0.5)
                    validation_losses.append(validate_model(i))
                #validation_losses = [validate_model(i) for i in dev_batch_start]
                this_validation_loss = numpy.mean(validation_losses)
                print('\t\tepoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in test_batch_start]
                    test_score = numpy.mean(test_losses)
                    print(('\t\t\t\tepoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,
                           test_score * 100.))
            '''

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


class Conv(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):
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
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
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
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input 
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = 1e-50+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = 1e-50+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

class Average_Pooling(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input, length_last_dim, kern):

        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        
        #para_tensor=self.W.dimshuffle('x', 'x', 0, 1)
        
        simi_tensor=compute_simi_feature(self.input, length_last_dim, self.W) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(T.sum(simi_tensor, axis=3).reshape((self.input.shape[0]/2, length_last_dim)),'simi_question')
        simi_answer=debug_print(T.sum(simi_tensor, axis=2).reshape((self.input.shape[0]/2, length_last_dim)), 'simi_answer')
        
        weights_question =T.nnet.softmax(simi_question) 
        weights_answer=T.nnet.softmax(simi_answer) 
        
        concate=T.concatenate([weights_question, weights_answer], axis=1)
        reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        weight_tensor=T.repeat(reshaped_concate, kern, axis=2)
        
        ele_add=self.input+weight_tensor
        self.output=T.sum(ele_add, axis=3)

        self.params = [self.W]

def compute_simi_feature(tensor, dim, para_matrix):
    odd_tensor=debug_print(tensor[0:tensor.shape[0]:2,:,:,:],'odd_tensor')
    even_tensor=debug_print(tensor[1:tensor.shape[0]:2,:,:,:], 'even_tensor')
    even_tensor_after_translate=debug_print(T.dot(para_matrix, even_tensor.reshape((tensor.shape[2], dim*tensor.shape[0]/2))), 'even_tensor_after_translate')
    fake_even_tensor=debug_print(even_tensor_after_translate.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3])),'fake_even_tensor')

    repeated_1=debug_print(T.repeat(odd_tensor, dim, axis=3),'repeated_1')
    repeated_2=debug_print(repeat_whole_matrix(fake_even_tensor, dim, False),'repeated_2')
    #repeated_2=T.repeat(even_tensor, even_tensor.shape[3], axis=2).reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3]**2))    
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=2)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=2)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=2),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    
    return list_of_simi.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[3], tensor.shape[3]))

def compute_acc(label_list, scores_list):
    #label_list contains 0/1, 500 as a minibatch, score_list contains score between -1 and 1, 500 as a minibatch
    if len(label_list)%500!=0 or len(scores_list)%500!=0:
        print 'len(label_list)%500: ', len(label_list)%500, ' len(scores_list)%500: ', len(scores_list)%500
        exit(0)
    if len(label_list)!=len(scores_list):
        print 'len(label_list)!=len(scores_list)', len(label_list), ' and ',len(scores_list)
        exit(0)
    correct_count=0
    total_examples=len(label_list)/500
    start_posi=range(total_examples)*500
    for i in start_posi:
        set_1=set()
        
        for scan in range(i, i+500):
            if label_list[scan]==1:
                set_1.add(scan)
        set_0=set(range(i, i+500))-set_1
        flag=True
        for zero_posi in set_0:
            for scan in set_1:
                if scores_list[zero_posi]> scores_list[scan]:
                    flag=False
        if flag==True:
            correct_count+=1
    
    return correct_count*1.0/total_examples
            
    
        


if __name__ == '__main__':
    evaluate_lenet5()
