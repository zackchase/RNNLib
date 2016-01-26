import numpy as np
import theano
import theano.tensor as T
from lstm import InputPLayer, SoftmaxPLayer, LSTMLayer, SoftmaxLayer
from lib import  floatX, make_caches, get_params, SGD, PerSGD, momentum, one_step_updates,random_weights, sigmoid

class PerRNN:
    def __init__(self, dnodex,inputdim,dim):
        X=T.ivector()
	Y=T.ivector()
	Z=T.lscalar()
	NP=T.lvector('NP')
        lambd = T.scalar()
	eta = T.scalar()
        temperature=T.scalar()
        num_input = inputdim
	self.umatrix=theano.shared(floatX(np.random.rand(dnodex.nuser,inputdim, inputdim)))
        self.pmatrix=theano.shared(floatX(np.random.rand(dnodex.npoi,inputdim)))
        self.B = theano.shared(np.zeros(dnodex.npoi).astype('float32'), name='B')
        self.p_l2_norm=(self.pmatrix**2).sum()
        self.u_l2_norm=(self.umatrix**2).sum()
        num_hidden = dim
        num_output = inputdim
        inputs = InputPLayer(self.pmatrix[X,:], self.umatrix[Z,:,:], name="inputs")
        lstm1 = LSTMLayer(num_input, num_hidden, input_layer=inputs, name="lstm1")
        #lstm2 = LSTMLayer(num_hidden, num_hidden, input_layer=lstm1, name="lstm2")
        #lstm3 = LSTMLayer(num_hidden, num_hidden, input_layer=lstm2, name="lstm3")
        softmax = SoftmaxPLayer(num_hidden, num_output, self.umatrix[Z,:,:], input_layer=lstm1, name="yhat", temperature=temperature)

        Y_hat = softmax.output()

        self.layers = inputs, lstm1,softmax
        params = get_params(self.layers)
        #caches = make_caches(params)

        cost = lambd*T.mean(T.nnet.categorical_crossentropy(Y_hat, T.dot(self.pmatrix[Y,:],self.umatrix[Z,:,:])))+lambd*self.p_l2_norm+lambd*self.u_l2_norm
    #    updates = PerSGD(cost,params,eta,X,Z,dnodex)#momentum(cost, params, caches, eta)
        updates = []
        grads = T.grad(cost=cost, wrt=params)
        updates.append([self.pmatrix,T.set_subtensor(self.pmatrix[X,:],self.pmatrix[X,:]-eta*grads[0])])
        updates.append([self.umatrix,T.set_subtensor(self.umatrix[Z,:,:],self.umatrix[Z,:,:]-eta*grads[1])])
        for p,g in zip(params[2:], grads[2:]):
            updates.append([p, p - eta * g])

        
        tmp_u=T.mean(T.dot(self.pmatrix[X,:],self.umatrix[Z,:,:]),axis=0)
        tr=T.dot(tmp_u,(self.pmatrix[X,:]-self.pmatrix[NP,:]).transpose())+self.B[X]-self.B[NP]
        pfp_loss=T.sum(T.log(T.nnet.sigmoid(tr))-lambd*(self.pmatrix[X]**2).sum(axis=1)-lambd*(self.umatrix[Z,:,:]**2).sum()-lambd*(self.pmatrix[NP]**2).sum(axis=1)-lambd*(self.umatrix[Z,:,:]**2).sum()- lambd* (self.B[X] ** 2 + self.B[NP] ** 2))
	pfp_cost=-pfp_loss
        g_cost_u = T.grad(cost=pfp_cost, wrt=self.umatrix)
        g_cost_p = T.grad(cost=pfp_cost, wrt=self.pmatrix)
        g_cost_B = T.grad(cost=pfp_cost, wrt=self.B)

        pfp_updates = [ (self.umatrix, self.umatrix - eta * g_cost_u), (self.pmatrix, self.pmatrix - eta * g_cost_p), (self.B, self.B - eta * g_cost_B) ]
        rlist=T.argsort(T.dot(tmp_u,self.pmatrix.T))[::-1]
       
        self.train = theano.function([X,Y,Z, eta, lambd, temperature], cost, updates=updates, allow_input_downcast=True)
        self.train_pfp=theano.function([X, NP, Z, eta, lambd], outputs=pfp_cost, updates=pfp_updates,allow_input_downcast=True, on_unused_input='ignore')
        self.predict_pfp = theano.function([X,Z], rlist, allow_input_downcast=True)

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
