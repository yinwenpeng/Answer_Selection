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
from loadData import load_msr_corpus_20161229, load_word2vec_to_init_new, load_mts, load_wmf_wikiQA, load_word2vec
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Average_Pooling_for_Top, create_conv_para, Diversify_Reg
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

from scipy import linalg, mat, dot

#need to modify
'''

1) rouge scores
2) data augment

'''
'''
0) word matching features
'''

def evaluate_lenet5(learning_rate=0.01, n_epochs=2000, nkerns=[400], batch_size=1, window_width=3,
                    maxSentLength=30, emb_size=300, hidden_size=[300,10],
                    margin=0.5, L2_weight=0.0001, Div_reg=0.0001, norm_threshold=5.0, use_svm=False):

    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/MicrosoftParaphrase/tokenized_msr/';
    rng = numpy.random.RandomState(23455)
    datasets, word2id=load_msr_corpus_20161229(rootPath+'tokenized_train.txt', rootPath+'tokenized_test.txt', maxSentLength)
    vocab_size=len(word2id)+1
    mtPath='/mounts/data/proj/wenpeng/Dataset/paraphraseMT/'
    mt_train, mt_test=load_mts(mtPath+'concate_15mt_train.txt', mtPath+'concate_15mt_test.txt')
    wm_train, wm_test=load_wmf_wikiQA(rootPath+'train_number_matching_scores.txt', rootPath+'test_number_matching_scores.txt')
    indices_train, trainY, trainLengths, normalized_train_length, trainLeftPad, trainRightPad= datasets[0]
    indices_train_l=indices_train[::2]
    indices_train_r=indices_train[1::2]
    trainLengths_l=trainLengths[::2]
    trainLengths_r=trainLengths[1::2]
    normalized_train_length_l=normalized_train_length[::2]
    normalized_train_length_r=normalized_train_length[1::2]

    trainLeftPad_l=trainLeftPad[::2]
    trainLeftPad_r=trainLeftPad[1::2]
    trainRightPad_l=trainRightPad[::2]
    trainRightPad_r=trainRightPad[1::2]    
    
    indices_test, testY, testLengths,normalized_test_length, testLeftPad, testRightPad= datasets[1]
    indices_test_l=indices_test[::2]
    indices_test_r=indices_test[1::2]
    testLengths_l=testLengths[::2]
    testLengths_r=testLengths[1::2]
    normalized_test_length_l=normalized_test_length[::2]
    normalized_test_length_r=normalized_test_length[1::2]
    
    testLeftPad_l=testLeftPad[::2]
    testLeftPad_r=testLeftPad[1::2]
    testRightPad_l=testRightPad[::2]
    testRightPad_r=testRightPad[1::2]  

    train_size = len(indices_train_l)
    test_size = len(indices_test_l)
    
    train_batch_start=range(train_size)
    test_batch_start=range(test_size)

    
#     indices_train_l=theano.shared(numpy.asarray(indices_train_l, dtype=theano.config.floatX), borrow=True)
#     indices_train_r=theano.shared(numpy.asarray(indices_train_r, dtype=theano.config.floatX), borrow=True)
#     indices_test_l=theano.shared(numpy.asarray(indices_test_l, dtype=theano.config.floatX), borrow=True)
#     indices_test_r=theano.shared(numpy.asarray(indices_test_r, dtype=theano.config.floatX), borrow=True)
#     indices_train_l=T.cast(indices_train_l, 'int32')
#     indices_train_r=T.cast(indices_train_r, 'int32')
#     indices_test_l=T.cast(indices_test_l, 'int32')
#     indices_test_r=T.cast(indices_test_r, 'int32')
    


    rand_values=random_value_normal((vocab_size, emb_size), theano.config.floatX, rng)
#     rand_values[0]=numpy.array(numpy.zeros(emb_size))
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init_new(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=numpy.array(rand_values,dtype=theano.config.floatX), borrow=True)#theano.shared(value=rand_values, borrow=True)      
    

    
    # allocate symbolic variables for the data
#     index = T.iscalar()
    x_index_l = T.imatrix()   # now, x is the index matrix, must be integer
    x_index_r = T.imatrix()
    y = T.ivector()  
    left_l=T.iscalar()
    right_l=T.iscalar()
    left_r=T.iscalar()
    right_r=T.iscalar()
    length_l=T.iscalar()
    length_r=T.iscalar()
    norm_length_l=T.fscalar()
    norm_length_r=T.fscalar()
    mts=T.fmatrix()
    wmf=T.fmatrix()
#     cost_tmp=T.fscalar()
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
    layer0_l_input = embeddings[x_index_l.flatten()].reshape((batch_size,maxSentLength, emb_size)).dimshuffle(0, 'x', 2,1)
    layer0_r_input = embeddings[x_index_r.flatten()].reshape((batch_size,maxSentLength, emb_size)).dimshuffle(0, 'x', 2,1)
    
    
    conv_W, conv_b=create_conv_para(rng, filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]))
    conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
    #layer0_output = debug_print(layer0.output, 'layer0.output')
    layer0_l = Conv_with_input_para(rng, input=layer0_l_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]), W=conv_W, b=conv_b)
    layer0_r = Conv_with_input_para(rng, input=layer0_r_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]), W=conv_W, b=conv_b)
    layer0_l_output=debug_print(layer0_l.output, 'layer0_l.output')
    layer0_r_output=debug_print(layer0_r.output, 'layer0_r.output')
    layer0_l_output_maxpool = T.max(layer0_l.output_narrow_conv_out[:,:,:,left_l:], axis=3).reshape((1, nkerns[0]))
    layer0_r_output_maxpool = T.max(layer0_r.output_narrow_conv_out[:,:,:,left_r:], axis=3).reshape((1, nkerns[0]))
    
    layer1=Average_Pooling_for_Top(rng, input_l=layer0_l_output, input_r=layer0_r_output, kern=nkerns[0],
                                       left_l=left_l, right_l=right_l, left_r=left_r, right_r=right_r, 
                                       length_l=length_l+filter_size[1]-1, length_r=length_r+filter_size[1]-1,
                                       dim=maxSentLength+filter_size[1]-1)
    

    
    
    
    
    
    sum_uni_l=T.sum(layer0_l_input[:,:,:,left_l:], axis=3).reshape((1, emb_size))
    norm_uni_l=sum_uni_l/T.sqrt((sum_uni_l**2).sum())
    sum_uni_r=T.sum(layer0_r_input[:,:,:,left_r:], axis=3).reshape((1, emb_size))
    norm_uni_r=sum_uni_r/T.sqrt((sum_uni_r**2).sum())
    
    uni_cosine=cosine(sum_uni_l, sum_uni_r)
    '''
    linear=Linear(sum_uni_l, sum_uni_r)
    poly=Poly(sum_uni_l, sum_uni_r)
    sigmoid=Sigmoid(sum_uni_l, sum_uni_r)
    rbf=RBF(sum_uni_l, sum_uni_r)
    gesd=GESD(sum_uni_l, sum_uni_r)
    '''
    eucli_1=1.0/(1.0+EUCLID(sum_uni_l, sum_uni_r))#25.2%
    #eucli_1=EUCLID(sum_uni_l, sum_uni_r)
    
    len_l=norm_length_l.reshape((1,1))
    len_r=norm_length_r.reshape((1,1))  
    
    '''
    len_l=length_l.reshape((1,1))
    len_r=length_r.reshape((1,1))  
    '''
    #length_gap=T.log(1+(T.sqrt((len_l-len_r)**2))).reshape((1,1))
    #length_gap=T.sqrt((len_l-len_r)**2)
    #layer3_input=mts
    HL_layer_1_input=T.concatenate([
#                                 mts, 
                                eucli_1, #uni_cosine,norm_uni_l-(norm_uni_l+norm_uni_r)/2,#uni_cosine, #
                                uni_cosine,
#                                 sum_uni_l,
#                                 sum_uni_r,
#                                 sum_uni_l+sum_uni_r,
                                1.0/(1.0+EUCLID(layer0_l_output_maxpool, layer0_r_output_maxpool)),
                                cosine(layer0_l_output_maxpool, layer0_r_output_maxpool),
                                layer0_l_output_maxpool,
                                layer0_r_output_maxpool,
                                T.sqrt((layer0_l_output_maxpool-layer0_r_output_maxpool)**2+1e-10),
                                
                                layer1.output_eucli_to_simi, #layer1.output_cosine,layer1.output_vector_l-(layer1.output_vector_l+layer1.output_vector_r)/2,#layer1.output_cosine, #
                                layer1.output_cosine,
                                layer1.output_vector_l,
                                layer1.output_vector_r,
                                T.sqrt((layer1.output_vector_l-layer1.output_vector_r)**2+1e-10),
#                                 len_l, len_r
                                layer1.output_attentions
#                                 wmf,
                                ], axis=1)#, layer2.output, layer1.output_cosine], axis=1)

    HL_layer_1_input_with_extra=T.concatenate([#HL_layer_1_input,
                                mts, len_l, len_r
#                                 wmf
                                ], axis=1)#, layer2.output, layer1.output_cosine], axis=1)

    HL_layer_1_input_size=1+1+   1+1+3* nkerns[0]   +1+1+3*nkerns[0]+10*10
    
    HL_layer_1_input_with_extra_size = HL_layer_1_input_size+15+2
    
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size[0], activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[1], activation=T.tanh)
    
    LR_layer_input=T.concatenate([HL_layer_2.output, HL_layer_1.output, HL_layer_1_input],axis=1)
    LR_layer_input_with_extra=T.concatenate([HL_layer_2.output,  HL_layer_1_input_with_extra],axis=1)#HL_layer_1.output,
    
    LR_layer=LogisticRegression(rng, input=LR_layer_input, n_in=HL_layer_1_input_size+hidden_size[0]+hidden_size[1], n_out=2)
#     LR_layer_input=HL_layer_2.output
#     LR_layer=LogisticRegression(rng, input=LR_layer_input, n_in=hidden_size, n_out=2)

#     layer3=LogisticRegression(rng, input=layer3_input, n_in=15+1+1+2+3, n_out=2)
    
    #L2_reg =(layer3.W** 2).sum()+(layer2.W** 2).sum()+(layer1.W** 2).sum()+(conv_W** 2).sum()
    L2_reg =debug_print((LR_layer.W** 2).sum()+(HL_layer_2.W** 2).sum()+(HL_layer_1.W** 2).sum()+(conv_W** 2).sum(), 'L2_reg')#+(layer1.W** 2).sum()
#     diversify_reg= Diversify_Reg(LR_layer.W.T)+Diversify_Reg(HL_layer_2.W.T)+Diversify_Reg(HL_layer_1.W.T)+Diversify_Reg(conv_W_into_matrix)
    cost_this =debug_print(LR_layer.negative_log_likelihood(y), 'cost_this')#+L2_weight*L2_reg
    cost=cost_this+L2_weight*L2_reg#+Div_reg*diversify_reg
    

    test_model = theano.function([x_index_l,x_index_r,y,left_l, right_l, left_r, right_r, length_l, length_r, norm_length_l, norm_length_r,
                                  mts,wmf], [LR_layer.errors(y), LR_layer.y_pred, LR_layer_input_with_extra, y], on_unused_input='ignore',allow_input_downcast=True)



    params = LR_layer.params+ HL_layer_2.params+HL_layer_1.params+[conv_W, conv_b]+[embeddings]#+[embeddings]# + layer1.params 
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        clipped_grad = T.clip(grad_i, -0.5, 0.5)
        acc = acc_i + T.sqr(clipped_grad)
        updates.append((param_i, param_i - learning_rate * clipped_grad / T.sqrt(acc+1e-10)))   #AdaGrad
        updates.append((acc_i, acc))    
  
    train_model = theano.function([x_index_l,x_index_r,y,left_l, right_l, left_r, right_r, length_l, length_r, norm_length_l, norm_length_r,
                                  mts,wmf], [cost,LR_layer.errors(y)], updates=updates, on_unused_input='ignore',allow_input_downcast=True)

    train_model_predict = theano.function([x_index_l,x_index_r,y,left_l, right_l, left_r, right_r, length_l, length_r, norm_length_l, norm_length_r,
                                  mts,wmf], [cost_this,LR_layer.errors(y), LR_layer_input_with_extra, y],on_unused_input='ignore',allow_input_downcast=True)



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is


    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    
    max_acc=0.0
    nn_max_acc=0.0
    best_iter=0
    cost_tmp=0.0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        shuffle(train_batch_start)#shuffle training data

        for index in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * train_size + minibatch_index +1

            minibatch_index=minibatch_index+1

#             if iter%update_freq != 0:
#                 cost_ij, error_ij, layer3_input, y=train_model_predict(batch_start)
#                 #print 'cost_ij: ', cost_ij
#                 cost_tmp+=cost_ij
#                 error_sum+=error_ij
#             else:

            cost_i, error_i= train_model(indices_train_l[index: index + batch_size],
                                                              indices_train_r[index: index + batch_size],
                                                              trainY[index: index + batch_size],
                                                              trainLeftPad_l[index],
                                                              trainRightPad_l[index],
                                                              trainLeftPad_r[index],
                                                              trainRightPad_r[index],
                                                              trainLengths_l[index],
                                                              trainLengths_r[index],
                                                              normalized_train_length_l[index],
                                                              normalized_train_length_r[index],
                                                              mt_train[index: index + batch_size],
                                                              wm_train[index: index + batch_size])
            cost_tmp+=cost_i
            if iter < 6000 and iter %100 ==0:
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_tmp/iter)
            if iter >= 6000 and iter % 100 == 0:
#             if iter%100 ==0:
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_tmp/iter)
                test_losses=[]
                test_y=[]
                test_features=[]
                for index in test_batch_start:
                    test_loss, pred_y, layer3_input, y=test_model(indices_test_l[index: index + batch_size],
                                                                  indices_test_r[index: index + batch_size],
                                                                  testY[index: index + batch_size],
                                                                  testLeftPad_l[index],
                                                                  testRightPad_l[index],
                                                                  testLeftPad_r[index],
                                                                  testRightPad_r[index],
                                                                  testLengths_l[index],
                                                                  testLengths_r[index],
                                                                  normalized_test_length_l[index],
                                                                  normalized_test_length_r[index],
                                                                  mt_test[index: index + batch_size],
                                                                  wm_test[index: index + batch_size])
                    #test_losses = [test_model(i) for i in test_batch_start]
                    test_losses.append(test_loss)
                    test_y.append(y[0])
                    test_features.append(layer3_input[0])
                    #write_file.write(str(pred_y[0])+'\n')#+'\t'+str(testY[i].eval())+

                #write_file.close()
                test_score = numpy.mean(test_losses)
                test_acc = (1-test_score) * 100.
                if test_acc > nn_max_acc:
                    nn_max_acc = test_acc
                print '\t\t\tepoch:', epoch, 'iter:', iter, 'current acc:', test_acc, 'nn_max_acc:', nn_max_acc

                #now, see the results of svm
                if use_svm:
                    train_y=[]
                    train_features=[]
                    for index in train_batch_start: 
                        cost_ij, error_ij, layer3_input, y=train_model_predict(indices_train_l[index: index + batch_size],
                                                                  indices_train_r[index: index + batch_size],
                                                                  trainY[index: index + batch_size],
                                                                  trainLeftPad_l[index],
                                                                  trainRightPad_l[index],
                                                                  trainLeftPad_r[index],
                                                                  trainRightPad_r[index],
                                                                  trainLengths_l[index],
                                                                  trainLengths_r[index],
                                                                  normalized_train_length_l[index],
                                                                  normalized_train_length_r[index],
                                                                  mt_train[index: index + batch_size],
                                                                  wm_train[index: index + batch_size])
                        train_y.append(y[0])
                        train_features.append(layer3_input[0])
                        #write_feature.write(' '.join(map(str,layer3_input[0]))+'\n')
                    #write_feature.close()
     
                    clf = svm.SVC(kernel='linear')#OneVsRestClassifier(LinearSVC()) #linear 76.11%, poly 75.19, sigmoid 66.50, rbf 73.33
                    clf.fit(train_features, train_y)
                    results=clf.predict(test_features)
                    lr=LinearRegression().fit(train_features, train_y)
                    results_lr=lr.predict(test_features)
                    corr_count=0
                    corr_lr=0
                    test_size=len(test_y)
                    for i in range(test_size):
                        if results[i]==test_y[i]:
                            corr_count+=1
                        if numpy.absolute(results_lr[i]-test_y[i])<0.5:
                            corr_lr+=1
                    acc=corr_count*1.0/test_size
                    acc_lr=corr_lr*1.0/test_size
                    if acc > max_acc:
                        max_acc=acc
                        best_iter=iter
                    if acc_lr> max_acc:
                        max_acc=acc_lr
                        best_iter=iter
                    print '\t\t\t\tsvm acc: ', acc, 'LR acc: ', acc_lr, ' max acc: ',    max_acc , ' at iter: ', best_iter

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


def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum()+1e-10)
    norm_uni_r=T.sqrt((vec2**2).sum()+1e-10)
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r+1e-10), 'uni-cosine')
    return simi.reshape((1,1))    
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))    
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))    
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))    
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))    
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))   
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-10).reshape((1,1))
    


if __name__ == '__main__':
    evaluate_lenet5()