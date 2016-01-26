import numpy as np
import theano
import theano.tensor as T
from lstm import InputNPLayer, SoftmaxNPLayer, LSTMLayer, SoftmaxLayer
from lib import  floatX, make_caches, get_params, SGD, NPerSGD, momentum, one_step_updates,random_weights,sigmoid

class NonPerRNN:
    def __init__(self, dnodex,inputdim,dim):
        X=T.ivector()
	Y=T.ivector()
	Z=T.lscalar()
	user=T.lvector('u')
        NP=T.ivector()
	lambd = T.scalar()
	eta = T.scalar()
        temperature=T.scalar()
        num_input = inputdim
	self.umatrix=theano.shared(floatX(np.random.randn(*(dnodex.nuser, inputdim))), name='umatrix')
        self.pmatrix=theano.shared(floatX(np.random.randn(*(dnodex.npoi,inputdim))), name='pmatrix')
        self.p_l2_norm=(self.pmatrix**2).sum()
	self.B = theano.shared(np.zeros(dnodex.npoi).astype('float32'), name='B')
        self.u_l2_norm=(self.umatrix**2).sum()
        num_hidden = dim
        num_output = inputdim
        inputs = InputNPLayer(self.pmatrix[X,:], name="inputs")
        lstm1 = LSTMLayer(num_input, num_hidden, input_layer=inputs, name="lstm1")
        #lstm2 = LSTMLayer(num_hidden, num_hidden, input_layer=lstm1, name="lstm2")
        #lstm3 = LSTMLayer(num_hidden, num_hidden, input_layer=lstm2, name="lstm3")
        softmax = SoftmaxNPLayer(num_hidden, num_output, input_layer=lstm1, name="yhat", temperature=temperature)

        Y_hat = softmax.output()

        self.layers = inputs, lstm1,softmax
        params = get_params(self.layers)
        #caches = make_caches(params)

	cost = lambd*10*T.mean(T.nnet.categorical_crossentropy(Y_hat, self.pmatrix[Y,:]))+lambd*self.p_l2_norm+lambd*self.u_l2_norm
       # updates = NPerSGD(cost,params,eta,X)#momentum(cost, params, caches, eta)
        updates = []
        grads = T.grad(cost=cost, wrt=params)
        updates.append([self.pmatrix,T.set_subtensor(self.pmatrix[X,:],params[0]-eta*grads[0])])
        for p,g in zip(params[1:], grads[1:]):
            updates.append([p, p - eta * eta * g])


        x_ui = T.dot(self.umatrix[user], self.pmatrix[X].T).diagonal()
        x_uj = T.dot(self.umatrix[user], self.pmatrix[NP].T).diagonal()

        x_uij = self.B[X] - self.B[NP] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - lambd * (self.umatrix[user] ** 2).sum(axis=1) - lambd * (self.pmatrix[X] ** 2).sum(axis=1) - lambd * (self.pmatrix[NP] ** 2).sum(axis=1) - lambd * (self.B[X] ** 2 + self.B[NP] ** 2))
        cost_bpr = - obj

        g_cost_u = T.grad(cost=cost_bpr, wrt=self.umatrix)
        g_cost_p = T.grad(cost=cost_bpr, wrt=self.pmatrix)
        g_cost_B = T.grad(cost=cost_bpr, wrt=self.B)

        bpr_updates = [ (self.umatrix, self.umatrix - eta * g_cost_u), (self.pmatrix, self.pmatrix - eta * g_cost_p), (self.B, self.B - eta * g_cost_B) ]

        self.train = theano.function([X,Y, eta, lambd, temperature], cost, updates=updates, allow_input_downcast=True)
        self.train_bpr=theano.function([X,NP,user,eta,lambd], outputs=cost_bpr, updates=bpr_updates,allow_input_downcast=True)

        rlist=T.argsort(T.dot(self.umatrix[Z,:],self.pmatrix.T))[::-1]
        self.predict_bpr = theano.function([Z], rlist, allow_input_downcast=True)
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
