import time
import argparse
from char_rnn import CharRNN
from non_personalized_rnn import NonPerRNN
from personalize_bpr_rnn import PerRNN
from lib import one_hot, one_hot_to_string, floatX, random_weights,sigmoid
from lf import LF,PFP,BPR
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
#        self.test_track=self.split_data(self.split)
#        for tindex in self.test_track:
#            user=self.ptrack[tindex]
#            for x in self.plist[tindex]:
#                self.ratings[user].item_set[x]=-1
        self.split_data2(self.split)
        self.p_l2_norm=None
        self.u_l2_norm=None
        self.max_check_in=0
        for u in ratings:
            if self.max_check_in<len(ratings[u].item_set):
                self.max_check_in=len(ratings[u].item_set)
    
    def split_data(self, percentage):
        cur_test=0
        res=[]
        test_case=(1-percentage)*len(self.plist)
        #print 'Test case: ', test_case
        while test_case>=cur_test:
            #print cur_test
            for i in range(len(self.ptrack)):
                if random.random()>percentage and test_case>=cur_test:
                    res.append(i)
                    cur_test+=1
                if test_case<cur_test:
                    break
        return res
    def split_data2(self, split):
        cur_test=0
        res=[]
        test_case=(1-split)*self.rating_count
        while test_case>=cur_test:
            for x in self.ratings:
                for y in self.ratings[x].item_set:
                    if random.random()>split and test_case>=cur_test:
                        self.ratings[x].item_set[y]=-1
                        cur_test+=1
                    if test_case<cur_test:
                        break
        print cur_test, test_case
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
            if not line[0].isdigit():
                continue
            line=line[:-1]
            #print line
            tokens = line.split('\t')
            if len(tokens)==1:
                user=tokens[0]
                line=fi.next()
                if line[0]=='0':
                    continue
                user_hash[user]=user_hash.get(user, int(len(user_hash.keys())))
                user=user_hash[user]
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
#            if rating_count>5000:
#                break
        fi.close()
        sys.stdout.write("%s reading completed\n" % fi)
        dnodex=pdata(user_hash,vocab_hash,vocab_items,poi_list,poi_track,rating_count,split,inputdim)
        print 'Complete Loading...'
        print '#User: ', dnodex.nuser
        print '#POI: ',dnodex.npoi
        print '#Check-in: ', dnodex.rating_count
        print '#POI tracks: ', len(poi_track)
        return dnodex

def personalized_train(dnodex, eta, iters):
    p={}
    for x in range(len(dnodex.plist)):
        p[dnodex.ptrack[x]]=p.get(dnodex.ptrack[x],[])
        p[dnodex.ptrack[x]].append(x)
    for it in xrange(iters):
        for user in xrange(dnodex.nuser):
            if user not in p or len(p[user])==0:
                continue
            i=random.choice(p[user])
            if len(dnodex.plist[i])<2:
                continue
            X = dnodex.plist[i][:-1]
            Y = dnodex.plist[i][1:]
            NP=[]
            user=dnodex.ptrack[i]
            for pos_p in X:
                if dnodex.ratings[user].item_set[pos_p]<0:
                    X.remove(pos_p)
                    if pos_p in Y:
                        Y.remove(pos_p)
            if len(X)!=len(Y):
                Y=Y[1:]
	    while len(NP)!=len(X):
	        neg_poi=np.random.randint(dnodex.npoi)
                while neg_poi in dnodex.ratings[user].item_set:
                    neg_poi=np.random.randint(dnodex.npoi)
                NP.append(neg_poi)
            if len(NP)==0:
                continue
            lossf=str(rnn.train(X,Y,user, eta, 1.0)) 
        if it%10000==0:
            sys.stdout.write('\r%d iterations...'%(it))
            sys.stdout.flush()

def personalized_pmf_train(dnodex, eta, iters):
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

def bpr_train(dnodex, eta, iters):
    p={}
    for x in range(len(dnodex.plist)):
        p[dnodex.ptrack[x]]=p.get(dnodex.ptrack[x],[])
        p[dnodex.ptrack[x]].append(x)
    for it in xrange(iters):
        for user in xrange(dnodex.nuser):
            if user not in p or len(p[user])==0:
                continue
            i=random.choice(p[user])
            X = dnodex.plist[i]
            NP=[]
            user=dnodex.ptrack[i]
            for pos_p in X:
                if dnodex.ratings[user].item_set[pos_p]<0:
                    X.remove(pos_p)
            if len(X)==0:
                continue
	    while len(NP)!=len(X):
	        neg_poi=np.random.randint(dnodex.npoi)
                while neg_poi in dnodex.ratings[user].item_set:
                    neg_poi=np.random.randint(dnodex.npoi)
                NP.append(neg_poi)
            tmp_u= rnn.trainneg(X,NP,user,eta)
            pfp_loss= rnn.trainpos(X,NP,user,eta)
        if it%10000==0:
            sys.stdout.write('\r%d iterations...'%(it))
            sys.stdout.flush()

def non_personalized_bpr_train(dnodex, eta, iters):
    p={}
    for x in range(len(dnodex.plist)):
        p[dnodex.ptrack[x]]=p.get(dnodex.ptrack[x],[])
        p[dnodex.ptrack[x]].append(x)
    for it in xrange(iters):
        for user in xrange(dnodex.nuser):
            if user not in p or len(p[user])==0:
                continue
            i=random.choice(p[user])
            if len(dnodex.plist[i])<2:
                continue
            X = dnodex.plist[i][:-1]
            Y = dnodex.plist[i][1:]
            NP=[]
            user=dnodex.ptrack[i]
            for pos_p in X:
                if dnodex.ratings[user].item_set[pos_p]<0:
                    X.remove(pos_p)
                    if pos_p in Y:
                        Y.remove(pos_p)
            if len(X)==0:
                continue
            if len(X)!=len(Y):
                Y=Y[1:]
	    while len(NP)!=len(X):
	        neg_poi=np.random.randint(dnodex.npoi)
                while neg_poi in dnodex.ratings[user].item_set:
                    neg_poi=np.random.randint(dnodex.npoi)
                NP.append(neg_poi)
            if len(NP)==0:
                continue
            lossf=str(rnn.train(X,Y, eta, 1.0)) 
            tmp_u= rnn.trainneg(X,NP,user,eta)
            pfp_loss= rnn.trainpos(X,NP,user,eta)
        if it%10000==0:
            sys.stdout.write('\r%d iterations...'%(it))
            sys.stdout.flush()


def personalized_pfp_train(dnodex, eta, iters):
    p={}
    for x in range(len(dnodex.plist)):
        p[dnodex.ptrack[x]]=p.get(dnodex.ptrack[x],[])
        p[dnodex.ptrack[x]].append(x)
    for it in xrange(iters):
        for user in xrange(dnodex.nuser):
            if user not in p or len(p[user])==0:
                continue
            i=random.choice(p[user])
            if len(dnodex.plist[i])<2:
                continue
            X = dnodex.plist[i][:-1]
            Y = dnodex.plist[i][1:]
            NP=[]
            user=dnodex.ptrack[i]
            for pos_p in X:
                if dnodex.ratings[user].item_set[pos_p]<0:
                    X.remove(pos_p)
                    if pos_p in Y:
                        Y.remove(pos_p)
            if len(X)!=len(Y):
                Y=Y[1:]
	    while len(NP)!=len(X):
	        neg_poi=np.random.randint(dnodex.npoi)
                while neg_poi in dnodex.ratings[user].item_set:
                    neg_poi=np.random.randint(dnodex.npoi)
                NP.append(neg_poi)
            if len(NP)==0:
                continue
            lossf=str(rnn.train(X,Y,user, eta, 1.0)) 
            tmp_u= rnn.trainneg(X,NP,user,eta)
            pfp_loss= rnn.trainpos(X,NP,user,eta)
        if it%10000==0:
            sys.stdout.write('\r%d iterations...'%(it))
            sys.stdout.flush()



def non_personalized_train(dnodex, eta, iters):
    for it in xrange(iters):
        i = random.randint(0,len(dnodex.plist)-1)
        while len(dnodex.plist[i])<=2 or i in dnodex.test_track:
            i=random.randint(0,len(dnodex.plist)-1)
        X = dnodex.plist[i][:-1]
        Y = dnodex.plist[i][1:]
        lossf=str(rnn.train(one_hot(X,len(format(dnodex.npoi,'b'))), one_hot(Y,len(format(dnodex.npoi,'b'))), eta, 1.0))
        if it%500==0:
            print "iteration: %s, cost: %s" % (str(it), lossf)

def infer_stochastic(dnodex, rnn):
    precision=0
    test_case=0.0000001
    for test_index in dnodex.test_track:
        test=dnodex.plist[test_index]
        if test_index%100==0:
            sys.stdout.write('\r%d prediction finished, current precision: %4f,%f,%f...' % (test_index,precision/test_case, precision, test_case))
            sys.stdout.flush()
        if test_index>=500:
            break
        if len(test)==1:
            continue
        for index in range(len(test)-1):
            x = [one_hot([test[index]],len(format(dnodex.npoi,'b'))).flatten()]
            probs = rnn.predict_char(x, 1)
            p = np.asarray(probs[0], dtype="float64")
            p /= p.sum()
            sample = np.random.multinomial(len(p), p)
            res=one_hot_to_string(sample)
            if res==test[index+1]:
                precision+=1.0
            test_case+=1.0
            rnn.reset_state()
    print 'Precision: ', precision/test_case

def AUC_metric(dnodex, rnn):
    precision=0
    test_case=0.000001
    cumu_auc=0.0
    if 1 in dnodex.ratings:
        user=1
        candidate=range(dnodex.npoi)
        X=[]
        for p in dnodex.ratings[user].item_set:
            if dnodex.ratings[user].item_set[p]>0:
                X.append(p)
        if len(candidate)>0 and len(X)>0:
            tmp_u=T.mean(T.dot(dnodex.pmatrix[X,:],dnodex.umatrix[user,:,:]),axis=0)
            r=T.dot(tmp_u,dnodex.pmatrix[candidate,:].T).eval()
            hit=0
            num_correct_pairs=0
            res=np.argsort(r)[::-1]
            for i in range(len(res)):
                if res[i] in X:
                    hit+=1
                else:
                    num_correct_pairs+=hit
            cumu_auc+= (float)(num_correct_pairs)/((len(candidate) - len(X)) * len(X))
            precision+=len(set(res[:10])&set(X))
            test_case+=1
    return ( cumu_auc/test_case)


def infer_personalized(dnodex, rnn):
    precision=0
    test_case=0.000001
    cumu_auc=0.0
    for test_index in dnodex.test_track:
        test=dnodex.plist[test_index]
	tuser=dnodex.ptrack[test_index]
        if test_index%100==0:
            sys.stdout.write('\r%d prediction finished, current precision: %4f,%f,%f...' % (test_index,precision/test_case, precision, test_case))
            sys.stdout.flush()
        if test_index>=500:
            break
        if len(test)==1:
            continue
        for index in range(len(test)-1):
            x=test[index]#dnodex.pmatrix[test[index],:] #= [one_hot([test[index]],len(format(dnodex.npoi,'b'))).flatten()]
            probs = rnn.predict_char([x],tuser, 1)
            r=T.dot(probs,T.dot(dnodex.pmatrix, dnodex.umatrix[tuser,:,:]).transpose()).eval()
            hit=0
            num_correct_pairs=0
            res=np.argsort(r[0])[::-1]
            for i in range(len(res)):
                if res[i] in test:
                    hit+=1
                else:
                    num_correct_pairs+=hit
            cumu_auc+= (float)(num_correct_pairs)/((dnodex.npoi - len(test)) * len(test))
            if test[index+1] in res[0:10]:
                precision+=1.0
            test_case+=1.0
            rnn.reset_state()
    print 'Precision: ', precision/test_case
    print 'AUC: ', cumu_auc/test_case


def infer_bpr(dnodex):
    precision=0.0
    test_case=0.000001
    cumu_auc=0.0
    for user in range(dnodex.nuser):
        if user%20==0:
            sys.stdout.write('\r%d user prediction finished, p@10: %4f, AUC: %4f...' % (user,precision/(10*test_case),cumu_auc/test_case))
            sys.stdout.flush()
        X=[]
        test=[]
        for p in dnodex.ratings[user].item_set:
            if dnodex.ratings[user].item_set[p]>0:
                X.append(p)
            else:
                test.append(p)
        if len(X)>0 and len(test)>0:
	    hit=0
            num_correct_pairs=0
            r=rnn.predict_bpr(user)
	    res=[x1 for x1 in r if x1 not in X]
	    for i in range(len(res)):
                if res[i] in test:
                    hit+=1
                else:
                    num_correct_pairs+=hit
            cumu_auc+= (float)(num_correct_pairs)/((dnodex.npoi-len(X) - len(test)) * len(test))
            precision+=len(set(res[:10])&set(test))
            test_case+=1
    print '\nPrecision: ', precision/(10*test_case)
    print '\nAUC: ', cumu_auc/test_case



def infer_pfp(dnodex):
    precision=0.0
    test_case=0.000001
    cumu_auc=0.0
    for user in range(dnodex.nuser):
        if user%20==0:
            sys.stdout.write('\r%d user prediction finished, p@10: %4f, AUC: %4f...' % (user,precision/(10*test_case),cumu_auc/test_case))
            sys.stdout.flush()
        X=[]
        test=[]
        for p in dnodex.ratings[user].item_set:
            if dnodex.ratings[user].item_set[p]>0:
                X.append(p)
            else:
                test.append(p)
        if len(X)>0 and len(test)>0:
	    hit=0
            num_correct_pairs=0
            r=rnn.predict_pfp(X,user)
	    res=[x1 for x1 in r if x1 not in X]
	    for i in range(len(res)):
                if res[i] in test:
                    hit+=1
                else:
                    num_correct_pairs+=hit
            cumu_auc+= (float)(num_correct_pairs)/((dnodex.npoi-len(X) - len(test)) * len(test))
            precision+=len(set(res[:10])&set(test))
            test_case+=1
    print '\nPrecision: ', precision/(10*test_case)
    print '\nAUC: ', cumu_auc/test_case






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-split', help='Split for testing', dest='split', type=float,default=0.8)
    parser.add_argument('-module', help='Model module, 0: Non_personalized LSTM, 1: Personalized LSTM, 2: PFP-AUC+Personalized LSTM, 3: BPR+Non_personalized LSTM, 4: BPR', dest='module', default=0, type=int)
    parser.add_argument('-inputdim', help='Dimensionality of input poi', dest='inputdim', default=10, type=int)
    parser.add_argument('-dim', help='Dimensionality of hidden layers', dest='dim', default=10, type=int)
    parser.add_argument('-eta',dest='eta',default=0.002,type=float)
    parser.add_argument('-iters', dest='iters', default=800,type=int)
    args=parser.parse_args()
    data = load_poi_data(args.fi,args.split,args.inputdim)
    if args.module==0:
	print 'Non_personalized Seq Modeling'
        rnn=NonPerRNN(data,args.inputdim,args.dim)
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
        infer_pfp(data)
    elif args.module==2:
        print 'Personalized Seq + PFP Modeling'
        rnn=PerRNN(data,args.inputdim,args.dim)
        t1=time.time()
        personalized_pfp_train(data,args.eta,args.iters)
	t2=time.time()
        print 'Train completed'
        print t2-t1
        print 'Prediction starts...'
        infer_pfp(data)        
    elif args.module==3:
	print 'BPR+Non-personalized Modeling'
	rnn=NonPerRNN(data,args.inputdim,args.dim)
	non_personalized_bpr_train(data,args.eta,args.iters)
	print 'Train completed'
        print 'Prediction starts...'
        infer_bpr(data)       
    elif args.module==4:
        print 'BPR'
	rnn=NonPerRNN(data,args.inputdim,args.dim)
        bpr_train(data,args.eta,args.iters)
        print 'Train completed'
        print 'Prediction starts...'
        infer_bpr(data)
