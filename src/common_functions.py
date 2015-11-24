import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from cis.deep.utils.theano import debug_print
from WPDefined import repeat_whole_matrix, repeat_whole_tensor

def create_conv_para(rng, filter_shape):
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

class Conv_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

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
        left_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

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
        left_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

class Average_Pooling(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim, window_size, maxSentLength): # length_l, length_r: valid lengths after conv


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
        
        input_l_matrix=input_l.reshape((input_l.shape[2], input_l.shape[3]))
        input_l_matrix=input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)]
        input_r_matrix=input_r.reshape((input_r.shape[2], input_r.shape[3]))
        input_r_matrix=input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)]
        
        
        simi_tensor=compute_simi_feature_batch1(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(T.sum(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
        simi_answer=debug_print(T.sum(simi_tensor, axis=0).reshape((1, length_r)), 'simi_answer')
        
        #weights_question =T.nnet.softmax(simi_question) 
        #weights_answer=T.nnet.softmax(simi_answer) 
        weights_question =simi_question
        weights_answer=simi_answer        
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        weights_question_matrix=T.repeat(weights_question, kern, axis=0)
        weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
        weighted_matrix_l=input_l_matrix*weights_question_matrix # first add 1e-20 for each element to make non-zero input for weight gradient
        weighted_matrix_r=input_r_matrix*weights_answer_matrix
        '''
        #without attention
        weighted_matrix_l=input_l_matrix # first add 1e-20 for each element to make non-zero input for weight gradient
        weighted_matrix_r=input_r_matrix
        '''
        
        sub_tensor_list_l=[]
        for i in range(window_size):
            if i ==0:
                sub_tensor_list_l.append(weighted_matrix_l.dimshuffle(('x', 0, 1)))
            else:
                sub_tensor_list_l.append(T.concatenate([weighted_matrix_l[:, i:], weighted_matrix_l[:, :i]], axis=1).dimshuffle(('x', 0, 1)))
        sub_tensor_l=T.concatenate(sub_tensor_list_l, axis=0)
        average_pooled_matrix_l=T.sum(sub_tensor_l, axis=0)[:, :1-window_size]
        average_pooled_tensor_l=average_pooled_matrix_l.reshape((input_l.shape[0], input_l.shape[1], input_l.shape[2], length_l-window_size+1))
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((input_l.shape[0], input_l.shape[1], input_l.shape[2], left_l), dtype=theano.config.floatX)
        right_padding = T.zeros((input_l.shape[0], input_l.shape[1], input_l.shape[2], right_l), dtype=theano.config.floatX)
        
        self.output_tensor_l = T.concatenate([left_padding, average_pooled_tensor_l, right_padding], axis=3) 
        

        sub_tensor_list_r=[]
        for i in range(window_size):
            if i ==0:
                sub_tensor_list_r.append(weighted_matrix_r.dimshuffle(('x', 0, 1)))
            else:
                sub_tensor_list_r.append(T.concatenate([weighted_matrix_r[:, i:], weighted_matrix_r[:, :i]], axis=1).dimshuffle(('x', 0, 1)))
        sub_tensor_r=T.concatenate(sub_tensor_list_r, axis=0)
        average_pooled_matrix_r=T.sum(sub_tensor_r, axis=0)[:, :1-window_size]        
        average_pooled_tensor_r=average_pooled_matrix_r.reshape((input_r.shape[0], input_r.shape[1], input_r.shape[2], length_r-window_size+1))
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((input_r.shape[0], input_r.shape[1], input_r.shape[2], left_r), dtype=theano.config.floatX)
        right_padding = T.zeros((input_r.shape[0], input_r.shape[1], input_r.shape[2], right_r), dtype=theano.config.floatX)
        
        self.output_tensor_r = T.concatenate([left_padding, average_pooled_tensor_r, right_padding], axis=3) 


        dot_l=T.sum(weighted_matrix_l, axis=1) # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=T.sum(weighted_matrix_r, axis=1)        
        norm_l=T.sqrt((dot_l**2).sum())
        norm_r=T.sqrt((dot_r**2).sum())
        
        self.output_vector_l=(dot_l/norm_l).reshape((1, kern))
        self.output_vector_r=(dot_r/norm_r).reshape((1, kern))      
        self.output_concate=T.concatenate([dot_l, dot_r], axis=0).reshape((1, kern*2))
        self.output_cosine=(T.sum(dot_l*dot_r)/norm_l/norm_r).reshape((1,1))
        
        '''
        dot_l=T.sum(input_l_matrix, axis=1) # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=T.sum(input_r_matrix, axis=1)        
        '''
        self.output_eucli=debug_print(T.sqrt(T.sqr(dot_l-dot_r).sum()+1e-20).reshape((1,1)),'output_eucli')
        self.output_eucli_to_simi=1.0/(1.0+self.output_eucli)
        #self.output_simi=self.output_eucli
        
        

        self.params = [self.W]

class Average_Pooling_for_ARCII(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r): # length_l, length_r: valid lengths after conv
        
        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        #input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        #input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
    
        
        #with attention
        dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
        '''
        #without attention
        dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
        '''
        '''
        #with attention, then max pooling
        dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')          
        '''
        norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
        norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')
        
        self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, input_l.shape[2])),'output_vector_l')
        self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, input_r.shape[2])), 'output_vector_r')      
        self.output_concate=T.concatenate([dot_l, dot_r], axis=0).reshape((1, input_l.shape[2]*2))
        self.output_cosine=debug_print((T.sum(dot_l*dot_r)/norm_l/norm_r).reshape((1,1)),'output_cosine')

        self.output_eucli=debug_print(T.sqrt(T.sqr(dot_l-dot_r).sum()+1e-20).reshape((1,1)),'output_eucli')
        self.output_eucli_to_simi=1.0/(1.0+self.output_eucli)
        #self.output_eucli_to_simi_exp=1.0/T.exp(self.output_eucli) # not good
        #self.output_sigmoid_simi=debug_print(T.nnet.sigmoid(T.dot(dot_l/norm_l, (dot_r/norm_r).T)).reshape((1,1)),'output_sigmoid_simi')    


class Average_Pooling_for_Top(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim): # length_l, length_r: valid lengths after conv



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
        
        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
        
        
        simi_tensor=compute_simi_feature_batch1(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(T.sum(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
        simi_answer=debug_print(T.sum(simi_tensor, axis=0).reshape((1, length_r)), 'simi_answer')
        
        #weights_question =T.nnet.softmax(simi_question) 
        #weights_answer=T.nnet.softmax(simi_answer) 
        weights_question =simi_question
        weights_answer=simi_answer        
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        weights_question_matrix=T.repeat(weights_question, kern, axis=0)
        weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
        dot_l=debug_print(T.sum(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')      
        '''
        #without attention
        dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
        '''
        '''
        #with attention, then max pooling
        dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
        dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')          
        '''
        norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
        norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')
        
        self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, kern)),'output_vector_l')
        self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, kern)), 'output_vector_r')      
        self.output_concate=T.concatenate([dot_l, dot_r], axis=0).reshape((1, kern*2))
        self.output_cosine=debug_print((T.sum(dot_l*dot_r)/norm_l/norm_r).reshape((1,1)),'output_cosine')

        self.output_eucli=debug_print(T.sqrt(T.sqr(dot_l-dot_r).sum()+1e-20).reshape((1,1)),'output_eucli')
        self.output_eucli_to_simi=1.0/(1.0+self.output_eucli)
        #self.output_eucli_to_simi_exp=1.0/T.exp(self.output_eucli) # not good
        #self.output_sigmoid_simi=debug_print(T.nnet.sigmoid(T.dot(dot_l/norm_l, (dot_r/norm_r).T)).reshape((1,1)),'output_sigmoid_simi')    
        self.output_attentions=unify_eachone(simi_tensor, length_l, length_r, 7)
        

        self.params = [self.W]

def compute_simi_feature_batch1(input_l_matrix, input_r_matrix, length_l, length_r, para_matrix, dim):
    #matrix_r_after_translate=debug_print(T.dot(para_matrix, input_r_matrix), 'matrix_r_after_translate')
    matrix_r_after_translate=input_r_matrix

    repeated_1=debug_print(T.repeat(input_l_matrix, dim, axis=1)[:, : (length_l*length_r)],'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
    repeated_2=debug_print(repeat_whole_tensor(matrix_r_after_translate, dim, False)[:, : (length_l*length_r)],'repeated_2')
    '''
    #cosine attention   
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
    
    '''
    #euclid, effective for wikiQA
    gap=debug_print(repeated_1-repeated_2, 'gap')
    eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
    simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')
    
    
    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature(tensor, dim, para_matrix):
    odd_tensor=debug_print(tensor[0:tensor.shape[0]:2,:,:,:],'odd_tensor')
    even_tensor=debug_print(tensor[1:tensor.shape[0]:2,:,:,:], 'even_tensor')
    even_tensor_after_translate=debug_print(T.dot(para_matrix, 1e-20+even_tensor.reshape((tensor.shape[2], dim*tensor.shape[0]/2))), 'even_tensor_after_translate')
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
#def unify_eachone(tensor, left1, right1, left2, right2, dim, Np):
def unify_eachone(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]

    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2    

    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True) 
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)
    
    new_rows=T.maximum(sentlength_1, sentlength_1*repeat_row+extra_row)
    new_cols=T.maximum(sentlength_2, sentlength_2*repeat_col+extra_col)
    
    #core=debug_print(core_2[:,:, :new_rows, : new_cols],'core')
    new_matrix=debug_print(matrix_2[:new_rows,:new_cols], 'new_matrix')
    #determine x, y start positions
    size_row=new_rows/Np
    remain_row=new_rows%Np
    size_col=new_cols/Np
    remain_col=new_cols%Np
    
    xx=debug_print(T.concatenate([T.arange(Np-remain_row+1)*size_row, (Np-remain_row)*size_row+(T.arange(remain_row)+1)*(size_row+1)]),'xx')
    yy=debug_print(T.concatenate([T.arange(Np-remain_col+1)*size_col, (Np-remain_col)*size_col+(T.arange(remain_col)+1)*(size_col+1)]),'yy')
    
    list_of_maxs=[]
    for i in xrange(Np):
        for j in xrange(Np):
            region=debug_print(new_matrix[xx[i]:xx[i+1], yy[j]:yy[j+1]],'region')
            #maxvalue1=debug_print(T.max(region, axis=2), 'maxvalue1')
            maxvalue=debug_print(T.max(region).reshape((1,1)), 'maxvalue')
            list_of_maxs.append(maxvalue)
    

    all_max_value=T.concatenate(list_of_maxs, axis=1).reshape((1, Np**2))
    
    return all_max_value            


class Create_Attention_Input_Cnn(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, tensor_l, tensor_r, dim,kern, l_left_pad, l_right_pad, r_left_pad, r_right_pad): # length_l, length_r: valid lengths after conv
        #first reshape into matrix
        matrix_l=tensor_l.reshape((tensor_l.shape[2], tensor_l.shape[3]))
        matrix_r=tensor_r.reshape((tensor_r.shape[2], tensor_r.shape[3]))
        #start
        repeated_1=debug_print(T.repeat(matrix_l, dim, axis=1),'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
        repeated_2=debug_print(repeat_whole_tensor(matrix_r, dim, False),'repeated_2')
        '''
        #cosine attention   
        length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
        length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')
    
        multi=debug_print(repeated_1*repeated_2, 'multi')
        sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
        
        list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
        simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
        
        '''
        #euclid, effective for wikiQA
        gap=debug_print(repeated_1-repeated_2, 'gap')
        eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
        simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((dim, dim)), 'simi_matrix')
        W_bound = numpy.sqrt(6. / (dim + kern))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(kern, dim)),dtype=theano.config.floatX),borrow=True) #a weight matrix kern*kern
        matrix_l_attention=debug_print(T.dot(self.W, simi_matrix.T), 'matrix_l_attention')
        matrix_r_attention=debug_print(T.dot(self.W, simi_matrix), 'matrix_r_attention')
        #reset zero at both side
        left_zeros_l=T.set_subtensor(matrix_l_attention[:,:l_left_pad], T.zeros((matrix_l_attention.shape[0], l_left_pad), dtype=theano.config.floatX))
        right_zeros_l=T.set_subtensor(left_zeros_l[:,-l_right_pad:], T.zeros((matrix_l_attention.shape[0], l_right_pad), dtype=theano.config.floatX))
        left_zeros_r=T.set_subtensor(matrix_r_attention[:,:r_left_pad], T.zeros((matrix_r_attention.shape[0], r_left_pad), dtype=theano.config.floatX))
        right_zeros_r=T.set_subtensor(left_zeros_r[:,-r_right_pad:], T.zeros((matrix_r_attention.shape[0], r_right_pad), dtype=theano.config.floatX))       
        #combine with original input matrix
        self.new_tensor_l=T.concatenate([matrix_l,right_zeros_l], axis=0).reshape((tensor_l.shape[0], 2*tensor_l.shape[1], tensor_l.shape[2], tensor_l.shape[3])) 
        self.new_tensor_r=T.concatenate([matrix_r,right_zeros_r], axis=0).reshape((tensor_r.shape[0], 2*tensor_r.shape[1], tensor_r.shape[2], tensor_r.shape[3])) 
        
        self.params=[self.W]

    
    
    