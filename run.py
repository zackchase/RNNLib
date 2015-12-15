import argparse
from char_rnn import CharRNN
from lib import one_hot, one_hot_to_string, floatX
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
    def __init__(self,uhash,phash,plist,ptrack):
        self.uhash=uhash
        self.phash=phash
        self.plist=plist
        self.ptrack=ptrack

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

#rnn = CharRNN()
def load_poi_data(f,split):
    if True:
        vocab_items = {}
        vocab_4table=[]
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
#                print "u", user
                line=fi.next()
#                print line
                continue
            token=tokens[1]
            time=tokens[0].split("T")[0]
            rating=1
            if date!=time or user!=pre_user:
                if len(poi_per_track)!=0:
                    poi_track[len(poi_list)]=pre_user
                    poi_list.append(poi_per_track)
#                    print poi_per_track
                pre_user=user
                date=time
                poi_per_track=[]
            if token not in vocab_hash:
                vocab_hash[token] = vocab_hash.get(token,int(len(vocab_hash)))
                vocab_4table.append(VocabItem(token))
            token=vocab_hash[token]
            vocab_4table[token].count+=1
            poi_per_track.append(token)
            vocab_items[user]=vocab_items.get(user,VocabItem(user))
            vocab_items[user].item_set[token]=rating
            rating_count += 1
            if rating_count % 10000 == 0:
                sys.stdout.write("\rReading ratings %d" % rating_count)
                sys.stdout.flush()
            #if rating_count>10000:
            #    break
            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        fi.close()
        sys.stdout.write("%s reading completed\n" % fi)
          #self.bytes = fi.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.rating_count = rating_count           # Total number of words in train file
        self.user_count=len(user_hash.keys())
        self.item_count=len(vocab_hash.keys())
        self.poi_track=poi_track
        self.poi_list=poi_list
        self.vocab_4table=vocab_4table
        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file

#        self.__sort(min_count)
        print self.rating_count
        print self.rating_count*percentage
        self.split(percentage)
        #assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print 'Total user in training file: %d' % self.user_count
        print 'Total item in training file: %d' % self.item_count
        print 'Total rating in file: %d' % self.rating_count
        print 'Total POI tracking (day): %d' % len(self.poi_track.keys())
        print len(poi_list)
#        print 'Total raiting in testing set: %d' % self.rating_count-int(self.rating_count*percentage)
        #print 'Vocab size: %d' % len(self)

    def split(self, percentage):
        cur_test=0
        test_case=(1-percentage)*self.rating_count
        print 'Test case: ', test_case
        #print test_case
        for user in self.vocab_items.keys():
            if len(self.vocab_items[user].item_set.keys())<5:
                continue
            if cur_test>=test_case:
                break
            for item in self.vocab_items[user].item_set.keys():
                if cur_test<test_case and np.random.random()>percentage:
                    cur_test+=1
                    self.vocab_items[user].item_set[item]+=10
        for i in range(len(self.poi_track)):
            user=self.poi_track[i]
            for item in self.poi_list[i]:
                if self.vocab_items[user].item_set[item]>8:
                    self.poi_list[i].remove(item)




seq_len = 150

def train(text,eta, iters):
    for it in xrange(iters):
        i = random.randint(0, len(text)/seq_len)
        j = i * seq_len

        X = text[j:(j+seq_len)]
        Y = text[(j+1):(j+1+seq_len)]

        print "iteration: %s, cost: %s" % (str(it), str(rnn.train(one_hot(X), one_hot(Y), eta, 1.0)))


def infer_stochastic(rnn, k, temperature, start_char=" "):
    x = [one_hot(start_char).flatten()]

    for i in xrange(k):
        probs = rnn.predict_char(x, temperature)
        p = np.asarray(probs[0], dtype="float64")
        p /= p.sum()
        sample = np.random.multinomial(1, p)
        sys.stdout.write(one_hot_to_string(sample))
        x = [sample]

    rnn.reset_state()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-split', help='Split for testing', dest='split', type=float,default=0.8)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=10, type=int)
    parser.add_argument('-eta',dest='eta',default=0.01,type=float)
    parser.add_argument('-iters', dest='iters', default=800,type=int)
    args=parser.parse_args()
    data = load_data(args.fi,args.split)
    rnn=CharRNN()
    train(data,args.eta,args.iters)
    infer_stochastic(rnn,100,0.5)

