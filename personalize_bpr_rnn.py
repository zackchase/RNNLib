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
	NP=T.ivector()
	lambd = T.scalar()
	eta = T.scalar()
        temperature=T.scalar()
        num_input = inputdim
	self.umatrix=theano.shared(floatX(np.random.rand(dnodex.nuser,inputdim, inputdim)))
        self.pmatrix=theano.shared(floatX(np.random.rand(dnodex.npoi,inputdim)))
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

        tmp_u=T.mean(T.dot(self.pmatrix[X,:],self.umatrix[Z,:,:]),axis=0)
        tr=T.dot(tmp_u,(self.pmatrix[X,:]-self.pmatrix[NP,:]).transpose())
        pfp_loss1=sigmoid(tr)
        pfp_loss=pfp_loss1*(T.ones_like(pfp_loss1)-pfp_loss1)
        tmp_u1=T.reshape(T.repeat(tmp_u,X.shape[0]),(inputdim,X.shape[0])).T
        pfp_lossv=T.reshape(T.repeat(pfp_loss,inputdim),(inputdim,X.shape[0])).T
	cost = lambd*10*T.mean(T.nnet.categorical_crossentropy(Y_hat, T.dot(self.pmatrix[Y,:],self.umatrix[Z,:,:])))+lambd*self.p_l2_norm+lambd*self.u_l2_norm
    #    updates = PerSGD(cost,params,eta,X,Z,dnodex)#momentum(cost, params, caches, eta)
        updates = []
        grads = T.grad(cost=cost, wrt=params)
        updates.append([self.pmatrix,T.set_subtensor(self.pmatrix[X,:],self.pmatrix[X,:]-eta*grads[0])])
        updates.append([self.umatrix,T.set_subtensor(self.umatrix[Z,:,:],self.umatrix[Z,:,:]-eta*grads[1])])
        for p,g in zip(params[2:], grads[2:]):
            updates.append([p, p - eta * g])

        rlist=T.argsort(T.dot(tmp_u,self.pmatrix.T))[::-1]
        n_updates=[(self.pmatrix, T.set_subtensor(self.pmatrix[NP,:],self.pmatrix[NP,:]-eta*pfp_lossv*tmp_u1-eta*lambd*self.pmatrix[NP,:]))]
	p_updates=[(self.pmatrix, T.set_subtensor(self.pmatrix[X,:],self.pmatrix[X,:]+eta*pfp_lossv*tmp_u1-eta*lambd*self.pmatrix[X,:])),(self.umatrix, T.set_subtensor(self.umatrix[Z,:,:],self.umatrix[Z,:,:]+eta*T.mean(pfp_loss)*(T.reshape(tmp_u,(tmp_u.shape[0],1))*T.mean(self.pmatrix[X,:]-self.pmatrix[NP,:],axis=0)))-eta*lambd*self.umatrix[Z,:,:])]
        
        self.train = theano.function([X,Y,Z, eta, lambd, temperature], cost, updates=updates, allow_input_downcast=True)
        self.trainpos=theano.function([X,NP,Z,eta, lambd],tmp_u, updates=p_updates,allow_input_downcast=True)
        self.trainneg=theano.function([X,NP,Z,eta, lambd],T.mean(pfp_loss), updates=n_updates,allow_input_downcast=True)
        
        
        self.predict_pfp = theano.function([X,Z], rlist, allow_input_downcast=True)

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
