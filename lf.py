import numpy as np
import theano
import theano.tensor as T
from lib import zeros

class LF:
    def get_params(self):
        return self.params
    def load_model(self):
        return
    def updates(self):
        return []

class BPR(LF):
    def __init__(self, dnodex, name=""):
        X=T.ivector()
        Z=T.lscalar()
        tmp_t=T.mean(T.dot(dnodex.pmatrix[X,:],dnodex.umatrix[Z,:,:]),axis=0)
        print len(tmp_r.eval())
        neg_poi=np.random.randint(self.dnodex.npoi)
        while neg_poi in self.dnodex.ratings[user]:
            neg_poi=np.random.randint(self.dnodex.npoi)
        r=T.dot(T.dot(tmp_t,dnodex.umatrix[user,:,:].transpose()),(dnodex.pmatrix[poi]-dnodex.pmatrix[neg_poi]).transpose())
        self.likelihood=-T.nnet.sigmoid(r)
	

        
