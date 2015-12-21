import argparse
from char_rnn import CharRNN
from non_personalized_rnn import NonPerRNN
from personalize_rnn import PerRNN
from lib import one_hot, one_hot_to_string, floatX, random_weights
import numpy as np
import theano
import theano.tensor as T
import sys
import random
###############################
#
#  Prepare the data
#
###############################

# f = open("../data/reuters21578/reut2-002.sgm")
#f = open("../data/tinyshakespeare/input.txt")

class pdata(object):
    def __init__(self,uhash,phash,ratings,plist,ptrack,rating_count,split,inputdim):
        self.uhash=uhash
        self.phash=phash
        self.plist=plist
        self.ptrack=ptrack
        self.ratings=ratings
        self.nuser=len(uhash)
        self.npoi=len(phash)
        self.umatrix=None
	self.pmatrix=None
	self.rating_count=rating_count
        self.split=split
	self.inputdim=inputdim
        self.test_track=self.split_data(self.split)

    def split_data(self, percentage):
        cur_test=0
        res=[]
        test_case=(1-percentage)*len(self.plist)
        #print 'Test case: ', test_case
        while test_case>=cur_test:
            #print cur_test
            for i in range(len(self.ptrack)):
                if random.random()>percentage*0.8 and test_case>=cur_test:
                    res.append(i)
                    cur_test+=1
                if test_case<cur_test:
                    break
        return res

class Pnode:
    def __init__(self,pid):
        self.pid=pid
        self.count=0
        self.item_set={}
def load_data(f,split):
    f=open("./input.txt")
    text = f.read()
    f.close()
    return text

def load_poi_data(fi,split,inputdim):
    if True:
        vocab_items = {}
        user_hash={}
        vocab_hash = {}
        poi_track={}
        poi_list=[]
        rating_count = 0
        fi = open(fi, 'r')
        poi_per_track=[]
        pre_user=-1
        date='-1'
        for line in fi:
            line=line[:-1]
            tokens = line.split('\t')
            if len(tokens)==1:
                user=tokens[0]
                user_hash[user]=user_hash.get(user, int(len(user_hash)))
                user=user_hash[user]
                line=fi.next()
                continue
            token=tokens[1]
            time=tokens[0].split("T")[0]
            rating=1
            if date!=time or user!=pre_user:
                if len(poi_per_track)!=0:
                    poi_track[len(poi_list)]=pre_user
                    poi_list.append(poi_per_track)
                pre_user=user
                date=time
                poi_per_track=[]
            if token not in vocab_hash:
                vocab_hash[token] = vocab_hash.get(token,int(len(vocab_hash)))
            token=vocab_hash[token]
            poi_per_track.append(token)
            vocab_items[user]=vocab_items.get(user,Pnode(user))
            vocab_items[user].item_set[token]=vocab_items[user].item_set.get(token,int(0))
            vocab_items[user].item_set[token]+=1
            rating_count += 1
            if rating_count % 10000 == 0:
                sys.stdout.write("\rReading ratings %d" % rating_count)
                sys.stdout.flush()
            #if rating_count>10000:
            #    break
        fi.close()
        sys.stdout.write("%s reading completed\n" % fi)
        dnodex=pdata(user_hash,vocab_hash,vocab_items,poi_list,poi_track,rating_count,split,inputdim)
        print 'Complete Loading...'
        print '#User: ', dnodex.nuser
        print '#POI: ',dnodex.npoi
        print '#Check-in: ', dnodex.rating_count
        print '#POI tracks: ', len(poi_track)
        print '#Tracks for testing:, ', len(dnodex.test_track)
        return dnodex

def personalized_train(dnodex, eta, iters):
    for it in xrange(iters):
        i = random.randint(0,len(dnodex.plist)-1)
        while len(dnodex.plist[i])<=2 or i in dnodex.test_track:
            i=random.randint(0,len(dnodex.plist)-1)
#        print len(dnodex.plist[i])
        X = dnodex.plist[i][:-1]
        Y = dnodex.plist[i][1:]
	user=dnodex.ptrack[i]
        lossf=str(rnn.train(X,Y,user, eta, 1.0))
        if it%500==0:
            print "iteration: %s, cost: %s" % (str(it), lossf)
            #infer_stochastic(dnodex,rnn)




def non_personalized_train(dnodex, eta, iters):
    for it in xrange(iters):
        i = random.randint(0,len(dnodex.plist)-1)
        while len(dnodex.plist[i])<=2 or i in dnodex.test_track:
            i=random.randint(0,len(dnodex.plist)-1)
#        print len(dnodex.plist[i])
        X = dnodex.plist[i][:-1]
        Y = dnodex.plist[i][1:]
        lossf=str(rnn.train(one_hot(X,len(format(dnodex.npoi,'b'))), one_hot(Y,len(format(dnodex.npoi,'b'))), eta, 1.0))
        if it%500==0:
            print "iteration: %s, cost: %s" % (str(it), lossf)
            #infer_stochastic(dnodex,rnn)

def infer_stochastic(dnodex, rnn):
    precision=0
    test_case=0
    for test_index in dnodex.test_track:
        test=dnodex.plist[test_index]
        if len(test)==1:
            #precision+=1
            #test_case+=1
            continue
        for index in range(len(test)-1):
            x = [one_hot([test[index]],len(format(dnodex.npoi,'b'))).flatten()]
            probs = rnn.predict_char(x, 1)
            #print probs
            p = np.asarray(probs[0], dtype="float64")
            p /= p.sum()
            sample = np.random.multinomial(len(p), p)
            #print sample
            res=one_hot_to_string(sample)
            if res==test[index+1]:
                precision+=1.0
            test_case+=1.0
        rnn.reset_state()
    print 'Precision: ', precision/test_case

def infer_personalized(dnodex, rnn):
    precision=0
    test_case=0.000001
    for test_index in dnodex.test_track:
        test=dnodex.plist[test_index]
	tuser=dnodex.ptrack[test_index]
        if len(test)==1:
            #precision+=1
            #test_case+=1
            continue
        for index in range(len(test)-1):
            x=test[index]#dnodex.pmatrix[test[index],:] #= [one_hot([test[index]],len(format(dnodex.npoi,'b'))).flatten()]
            #print x, tuser
            probs = rnn.predict_char([x],tuser, 1)
            #print probs
            r=np.array(T.dot(probs,T.dot(dnodex.pmatrix, dnodex.umatrix[tuser,:,:]).transpose()).eval())
            res=np.argsort(r)[:11]
            #print res
            if test[index+1] in res:
                precision+=1.0
            test_case+=1.0
            rnn.reset_state()
    print 'Precision: ', precision/test_case




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-split', help='Split for testing', dest='split', type=float,default=0.8)
    parser.add_argument('-module', help='Model module, 0: non_personalized, 1: personalized, 2: collaborative, 3: MF+collaborative+personalized', dest='module', default=0, type=int)
    parser.add_argument('-inputdim', help='Dimensionality of input poi', dest='inputdim', default=10, type=int)
    parser.add_argument('-dim', help='Dimensionality of hidden layers', dest='dim', default=10, type=int)
    parser.add_argument('-eta',dest='eta',default=0.002,type=float)
    parser.add_argument('-iters', dest='iters', default=800,type=int)
    args=parser.parse_args()
    data = load_poi_data(args.fi,args.split,args.inputdim)
    if args.module==0:
	print 'Non_personalized Seq Modeling'
        rnn=NonPerRNN(data,args.dim)
    	non_personalized_train(data,args.eta,args.iters)
	print 'Train completed'
        print 'Prediction starts...'
    	infer_stochastic(data,rnn)
    elif args.module==1:
        print 'Personalized Seq Modeling'
        rnn=PerRNN(data,args.inputdim,args.dim)
	personalized_train(data,args.eta,args.iters)
	print 'Train completed'
        print 'Prediction starts...'
        infer_personalized(data, rnn)
	

