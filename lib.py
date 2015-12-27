import numpy as np
import theano
import theano.tensor as T

def one_hot(seq,s):
    """
    Take a seq and size of total items,s as input, return one-hot-bit representation of the seq
    """
    res = np.zeros((len(seq), s))
    for i in xrange(len(seq)):
	tmpr=format(seq[i],'b')
	for x in range(len(tmpr)):
            res[i,s-1-x] = float(tmpr[len(tmpr)-1-x])+0.
    return res

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def random_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def zeros(shape):
    return theano.shared(floatX(np.zeros(shape)))

def softmax(X, temperature=1.0):
    e_x = T.exp((X - X.max(axis=1).dimshuffle(0, 'x'))/temperature)
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# def softmax(X, temperature):
#     e_x = T.exp((X-X.max())/temperature)
#     return e_x / e_x.sum()

def sigmoid(X):
    return 1 / (1 + T.exp(-X))


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X


def rectify(X):
    return T.maximum(X, 0.)


def SGD (cost, params, eta):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p,g in zip(params, grads):
        updates.append([p, p - eta * g])

    return updates
def PerSGD (cost, params, eta, X, Z, dnodex):
    updates = []
    grads = T.grad(cost=cost, wrt=params)
    updates.append([dnodex.pmatrix,T.set_subtensor(dnodex.pmatrix[X,:],dnodex.pmatrix[X,:]-eta*eta*grads[0])])
    updates.append([dnodex.umatrix,T.set_subtensor(dnodex.umatrix[Z,:,:],dnodex.umatrix[Z,:,:]-eta*eta*grads[1])])
    for p,g in zip(params[2:], grads[2:]):
        updates.append([p, p + eta * eta * g])

    return updates



def momentum(cost, params, caches, eta, rho=.1):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p, c, g in zip(params, caches, grads):
        delta = rho * g + (1-rho) * c
        updates.append([c, delta])
        updates.append([p, p - eta * delta])

    return updates


def sample_char(probs):
    return one_hot_to_string(np.random.multinomial(1, probs))


def one_hot_to_string(one_hot):
    return int(''.join('1' if x>0 else '0' for x in one_hot),2)

def get_params(layers):
    params = []
    for layer in layers:
        params += layer.get_params()
    return params

def zeros(length):
    return theano.shared(floatX(np.zeros(length)))

def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))

    return caches

def one_step_updates(layers):
    updates = []

    for layer in layers:
        updates += layer.updates()

    return updates
