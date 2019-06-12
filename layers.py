from inits import *
import tensorflow as tf
import numpy as np
import sys as sys
import datetime
np.set_printoptions(threshold=sys.maxsize)

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            print(inputs)
            print(outputs)
#            if self.logging:
#                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.placeholders=placeholders

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        x = inputs
        self.support = self.placeholders['support']
        
        print( "self.support[0].shape:----------")
        print( self.support[0].shape)
        
        
        
        '''
        ## support === adjacency matrix
        ## inputs  === features
        '''
        
        #print("x.shape:")
        #print(x.shape)
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # NO POOLING #####
        supports = list()
        #print("len(self.support)="+str(len(self.support)))
        for i in range(len(self.support)):
            
            #print("self.vars['weights_' + str(i)].shape="+str(self.vars['weights_' + str(i)].shape))
            if not self.featureless:
                print("not self.featureless")
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
                print("pre_sup.shape="+str(pre_sup.shape))
            else:
                pre_sup = self.vars['weights_' + str(i)]
                print("pre_sup.shape="+str(pre_sup.shape))
            print('self.support['+str(i)+'].shape'+str(self.support[i].shape))
            print('pre_sup.shape'+str(pre_sup.shape))
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        #print("output.shape="+str(output.shape))

        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output #output
        return self.act(output)

class GraphMaxPooling(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim,placeholders, dropout=True, sparse_inputs=True,**kwargs):
        super(GraphMaxPooling, self).__init__(**kwargs)
        self.input_dim=input_dim
        
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.support = placeholders['support']
        self.placeholders=placeholders
   #     print("len(self.support)")
       # print(len(self.support))
      #  print("self.support[0].shape")
        #print(self.support[0].shape)
        self.sparse_inputs = sparse_inputs

        self.num_features_nonzero = placeholders['num_features_nonzero']

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # inputs are not used
  #      print("GraphMaxPooling._call")
        x = inputs
        ## support === ADJACENCY MATRIX
        supports = list()
        for support_no in range(len(self.support)):
            print("self.support[support_no]")
            print(self.support[support_no])
            support=tf.sparse_tensor_to_dense(self.support[support_no])
            
            rightMatrixToDeleteColumnsWithEvenIndices=np.eye(self.input_dim,dtype=np.float32)
            rightMatrixToDeleteColumnsWithEvenIndices=rightMatrixToDeleteColumnsWithEvenIndices[:, range(1,self.input_dim,2)]
 #           print("rightMatrixToDeleteColumnsWithEvenIndices.shape="+str(rightMatrixToDeleteColumnsWithEvenIndices.shape)+"  "+str(datetime.datetime.now()))
 #           print(rightMatrixToDeleteColumnsWithEvenIndices.dtype)
            leftMatrixToDeleteRowsWithEvenIndices=rightMatrixToDeleteColumnsWithEvenIndices.T
#            print(leftMatrixToDeleteRowsWithEvenIndices.dtype)
            coarse_support=tf.matmul(support,rightMatrixToDeleteColumnsWithEvenIndices)
            coarse_support=tf.matmul(leftMatrixToDeleteRowsWithEvenIndices,coarse_support)
 #           print(coarse_support)
            supports.append(tf.contrib.layers.dense_to_sparse(coarse_support))
 #           print("supports="+str(supports)+str(datetime.datetime.now()))
        #output = tf.add_n(supports)
        self.placeholders['support'][0]=supports[0]
        output = supports[0]
#        print("OUTPUT:")
#        print(output)
        self.embedding =output #output
#        print("output.shape="+str(output.shape))

        return output
