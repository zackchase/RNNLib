import numpy as np
import theano
import theano.tensor as T
from lib import softmax, dropout, floatX, random_weights, zeros


class NNLayer:

    def get_params(self):
        return self.params

    def save_model(self):
        return

    def load_model(self):
        return

    def updates(self):
        return []

    def reset_state(self):
        return

class LSTMLayer(NNLayer):

    def __init__(self, num_input, num_cells, input_layer=None, name=""):
        """
        LSTM Layer
        Takes as input sequence of inputs, returns sequence of outputs
        """

        self.name = name
        self.num_input = num_input
        self.num_cells = num_cells

        self.X = input_layer.output()
        self.h0 = theano.shared(floatX(np.zeros(num_cells)))
        self.s0 = theano.shared(floatX(np.zeros(num_cells)))

        self.W_gx = random_weights((num_input, num_cells))
        self.W_ix = random_weights((num_input, num_cells))
        self.W_fx = random_weights((num_input, num_cells))
        self.W_ox = random_weights((num_input, num_cells))

        self.W_gh = random_weights((num_cells, num_cells))
        self.W_ih = random_weights((num_cells, num_cells))
        self.W_fh = random_weights((num_cells, num_cells))
        self.W_oh = random_weights((num_cells, num_cells))

        self.b_g = zeros(num_cells)
        self.b_i = zeros(num_cells)
        self.b_f = zeros(num_cells)
        self.b_o = zeros(num_cells)

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                        self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                        self.b_g, self.b_i, self.b_f, self.b_o,
                ]


    def one_step(self, x, h_tm1, s_tm1):
        """
        """
        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i*g + s_tm1 * f
        h = T.tanh(s) * o

        return h, s

    def output(self):

        ([outputs, states], updates) = theano.scan(
                fn=self.one_step,
                sequences=self.X,
                outputs_info = [self.h0, self.s0],
            )

        self.new_s = states[-1]
        self.new_h  = outputs[-1]

        return outputs

    def updates(self):
        return [(self.s0, self.new_s), (self.h0, self.new_h)]

    def reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_cells)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_cells)))


class FullyConnectedLayer(NNLayer):
    """
    """
    def __init__self(self, num_input, num_output, input_layer, name=""):

        self.X = input_layer.output()
        self.num_input = num_input
        self.num_output = num_output

        self.W = random_weights((num_input, num_output))
        self.b = zeros(num_output)

        self.params = [self.W, self.B]

    def output(self):
        return

class InputNPLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.params=[]
	self.X = X#theano.shared(X)
        self.params=[X]
    def output(self):
        return self.X


class InputPLayer(NNLayer):
    """
    """
    def __init__(self, X, Z, name=""):
        self.name = name
        self.params=[]
	self.X = X#theano.shared(X)
	self.Z = Z#theano.shared(Z)#dnodex.umatrix[0,:,:]
        self.params=[X,Z]
    def output(self):
        return T.dot(self.X,self.Z)



class InputLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params=[]

    def output(self):
        return self.X

class SoftmaxNPLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layer, temperature=1.0, name=""):
        self.name = ""
        self.X = input_layer.output()
        self.params = []
        self.temp = temperature
        self.W_yh = random_weights((num_input, num_output))
        self.b_y = zeros(num_output)

        self.params = [self.W_yh, self.b_y]

    def output(self):
        return softmax((T.dot(self.X, self.W_yh) + self.b_y), temperature=self.temp)



class SoftmaxPLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, Z, input_layer, temperature=1.0, name=""):
        self.name = ""
        self.X = input_layer.output()
        self.params = []
        self.temp = temperature
	self.Z=Z
        self.W_yh = random_weights((num_input, num_output))
        self.b_y = zeros(num_output)

        self.params = [self.W_yh, self.b_y]

    def output(self):
        return softmax((T.dot(T.dot(self.X, self.W_yh) + self.b_y, self.Z)), temperature=self.temp)


class SoftmaxLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layer, temperature=1.0, name=""):
        self.name = ""
        self.X = input_layer.output()
        self.params = []
        self.temp = temperature

        self.W_yh = random_weights((num_input, num_output))
        self.b_y = zeros(num_output)

        self.params = [self.W_yh, self.b_y]

    def output(self):
        return softmax((T.dot(self.X, self.W_yh) + self.b_y), temperature=self.temp)


class SigmoidLayer(NNLayer):

    def __init__(self, input_layer, name=""):
        self.X = input_layer.output()
        self.params = []

    def output(self):
        return sigmoid(self.X)

class DropoutLayer(NNLayer):

    def __init__(self, input_layer, name=""):
        self.X = input_layer.output()
        self.params = []

    def output(self):
        return dropout(self.X)

class MergeLayer(NNLayer):
    def init(self, input_layers):
        return

    def output(self):
        return




