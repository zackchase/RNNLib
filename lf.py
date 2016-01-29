import numpy as np
import theano
import theano.tensor as T
from lib import zeros,floatX

class LF:
    def get_params(self):
        return self.params
    def load_model(self):
        return
    def updates(self):
        return []

class BPR(LF):
    def __init__(self, dnodex,inputdim, name=""):
        pos_p=T.lscalar()
        neg_poi=T.lscalar()
	user=T.lscalar()
	eta=T.scalar()
	pfp_loss=T.scalar()
	if dnodex.pmatrix is None:
	    dnodex.umatrix=theano.shared(floatX(np.random.randn(*(dnodex.nuser, inputdim))))
            dnodex.pmatrix=theano.shared(floatX(np.random.randn(*(dnodex.npoi,inputdim))))
	n_updates=[(dnodex.pmatrix, T.set_subtensor(dnodex.pmatrix[neg_poi,:],dnodex.pmatrix[neg_poi,:]-eta*pfp_loss*dnodex.umatrix[user,:]-eta*eta*dnodex.pmatrix[neg_poi,:]))]
        p_updates=[(dnodex.pmatrix, T.set_subtensor(dnodex.pmatrix[pos_p,:],dnodex.pmatrix[pos_p,:]+eta*pfp_loss*dnodex.umatrix[user,:]-eta*eta*dnodex.pmatrix[pos_p,:])),(dnodex.umatrix, T.set_subtensor(dnodex.umatrix[user,:],dnodex.umatrix[user,:]+eta*pfp_loss*(dnodex.pmatrix[pos_p,:]-dnodex.pmatrix[neg_poi,:])-eta*eta*dnodex.umatrix[user,:]))]
        self.trainpos=theano.function([pos_p,neg_poi,user,eta,pfp_loss],updates=p_updates,allow_input_downcast=True)
        self.trainneg=theano.function([neg_poi,user,eta,pfp_loss],updates=n_updates,allow_input_downcast=True)
        
class PFP(LF):
    def __init__(self, dnodex, inputdim,name=""):
        pos_p=T.lscalar()
        neg_poi=T.lscalar()
	user=T.lscalar()
	eta=T.scalar()
	pfp_loss=T.scalar()
        X=T.ivector()
	if dnodex.pmatrix is None:
	    dnodex.umatrix=theano.shared(floatX(np.random.randn(*(dnodex.nuser, inputdim))))
            dnodex.pmatrix=theano.shared(floatX(np.random.randn(*(dnodex.npoi,inputdim))))
	tmp_u=T.mean(T.dot(dnodex.pmatrix[X,:],dnodex.umatrix[user,:,:]),axis=0)
        tmp_p=T.mean(dnodex.pmatrix[X,:],axis=0)
	n_updates=[(dnodex.pmatrix, T.set_subtensor(dnodex.pmatrix[neg_poi,:],dnodex.pmatrix[neg_poi,:]-eta*pfp_loss*tmp_u-eta*eta*dnodex.pmatrix[neg_poi,:]))]
	p_updates=[(dnodex.pmatrix, T.set_subtensor(dnodex.pmatrix[pos_p,:],dnodex.pmatrix[pos_p,:]+eta*pfp_loss*tmp_u-eta*eta*dnodex.pmatrix[pos_p,:])),(dnodex.umatrix, T.set_subtensor(dnodex.umatrix[user,:,:],dnodex.umatrix[user,:,:]+eta*pfp_loss*T.dot(tmp_p.T,dnodex.pmatrix[pos_p,:]-dnodex.pmatrix[neg_poi,:])-eta*eta*dnodex.umatrix[user,:,:]))]
        self.trainpos=theano.function([pos_p,neg_poi,user,eta,pfp_loss,X],updates=p_updates,allow_input_downcast=True)
        self.trainneg=theano.function([neg_poi,user,eta,pfp_loss,X],updates=n_updates,allow_input_downcast=True)
        
