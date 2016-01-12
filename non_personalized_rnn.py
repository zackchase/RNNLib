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
	NP=T.ivector()
	eta = T.scalar()
        temperature=T.scalar()
        num_input = inputdim
	dnodex.umatrix=theano.shared(floatX(np.random.randn(*(dnodex.nuser, inputdim))))
        dnodex.pmatrix=theano.shared(floatX(np.random.randn(*(dnodex.npoi,inputdim))))
        dnodex.p_l2_norm=(dnodex.pmatrix**2).sum()
        dnodex.u_l2_norm=(dnodex.umatrix**2).sum()
        num_hidden = dim
        num_output = inputdim
        inputs = InputNPLayer(dnodex.pmatrix[X,:], name="inputs")
        lstm1 = LSTMLayer(num_input, num_hidden, input_layer=inputs, name="lstm1")
        #lstm2 = LSTMLayer(num_hidden, num_hidden, input_layer=lstm1, name="lstm2")
        #lstm3 = LSTMLayer(num_hidden, num_hidden, input_layer=lstm2, name="lstm3")
        softmax = SoftmaxNPLayer(num_hidden, num_output, input_layer=lstm1, name="yhat", temperature=temperature)

        Y_hat = softmax.output()

        self.layers = inputs, lstm1,softmax
        params = get_params(self.layers)
        #caches = make_caches(params)

	tr=T.dot(dnodex.umatrix[Z,:],(dnodex.pmatrix[X,:]-dnodex.pmatrix[NP,:]).transpose())
        pfp_loss1=sigmoid(tr)
        pfp_loss=pfp_loss1*(T.ones_like(pfp_loss1)-pfp_loss1)
	tmp_u1=T.reshape(T.repeat(dnodex.umatrix[Z,:],X.shape[0]),(inputdim,X.shape[0])).T
        pfp_lossv=T.reshape(T.repeat(pfp_loss,inputdim),(inputdim,X.shape[0])).T
	cost = T.mean(T.nnet.categorical_crossentropy(Y_hat, dnodex.pmatrix[Y,:]))+eta*dnodex.p_l2_norm+eta*dnodex.u_l2_norm
        updates = NPerSGD(cost,params,eta,X,dnodex)#momentum(cost, params, caches, eta)

        n_updates=[(dnodex.pmatrix, T.set_subtensor(dnodex.pmatrix[NP,:],dnodex.pmatrix[NP,:]-eta*pfp_lossv*tmp_u1-eta*eta*dnodex.pmatrix[NP,:]))]
	p_updates=[(dnodex.pmatrix, T.set_subtensor(dnodex.pmatrix[X,:],dnodex.pmatrix[X,:]+eta*pfp_lossv*tmp_u1-eta*eta*dnodex.pmatrix[X,:]))]
        self.train = theano.function([X,Y, eta, temperature], cost, updates=updates, allow_input_downcast=True)
        self.trainpos=theano.function([X,NP,Z,eta],T.mean(pfp_loss), updates=p_updates,allow_input_downcast=True)
        self.trainneg=theano.function([X,NP,Z,eta],T.mean(pfp_loss), updates=n_updates,allow_input_downcast=True)

	rlist=T.argsort(T.dot(dnodex.umatrix[Z,:],dnodex.pmatrix.T))[::-1]
        self.predict_bpr = theano.function([Z], rlist, allow_input_downcast=True)
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
