import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/thangbui/synced/sandbox/neural/temp/'
srng = RandomStreams()
print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))


def rectify(X):
    return T.switch(X<0, 0, X)

def sigmoid(X):
    return 1 / (1 + T.exp(-X))

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def model_sigmoid(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = sigmoid(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = sigmoid(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = T.dot(h2, w_o)
    return h, h2, py_x


def model_rectify(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = T.dot(h2, w_o)
    return h, h2, py_x

def get_approx_Hessian_difference():
    return 0 # TODO

def get_approx_Hessian_outerprod():
    return 0 # TODO

def get_approx_Hessian_Rbackprop():
    return 0 # TODO

def get_sensitivity_matrix():
    return 0 # TODO

# generate some training data
trX = np.linspace(-1, 1, 101)
trY = (trX-0.5)**3 + np.sin(5*trX) + 0.2*np.random.randn(101,)

trX = trX.reshape(len(trX), 1)
trY = trY.reshape(len(trY), 1)

teX = np.linspace(-2, 2, 201)
teX = teX.reshape(len(teX), 1)

# create theano variables
X = T.fmatrix()
Y = T.fmatrix()

# generate theano functions
w_h = init_weights((1, 50))
w_h2 = init_weights((50, 50))
w_o = init_weights((50, 1))

h, h2, py_x_dropout = model_sigmoid(X, w_h, w_h2, w_o, 0., 0.1)
h, h2, py_x = model_sigmoid(X, w_h, w_h2, w_o, 0., 0)
# h, h2, py_x = model_rectify(X, w_h, w_h2, w_o, 0.1, 0.1)

cost_dropout = T.mean(T.sqr(py_x_dropout - Y))
params = [w_h, w_h2, w_o]
updates_dropout = RMSprop(cost_dropout, params, lr=0.01)

train_dropout = theano.function(inputs=[X, Y], outputs=cost_dropout,
                                updates=updates_dropout, allow_input_downcast=True)

predict_dropout = theano.function(inputs=[X], outputs=py_x_dropout,
                                  allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x,
                          allow_input_downcast=True)

for i in range(5000):
    cost = train_dropout(trX, trY)
    print i, np.mean(trY - predict(np.array(trX)))

fig = plt.figure()
for i in range(40):
    plt.plot(teX, predict_dropout(np.array(teX)), 'b-')
plt.plot(trX, trY, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('train using dropout, prediction using dropout')
fig.savefig(path + 'dropout.png')

fig = plt.figure()
for i in range(40):
    cost = train_dropout(trX, trY)
    plt.plot(teX, predict(np.array(teX)), 'b-')
plt.plot(trX, trY, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('train with dropout, prediction using last 40 SGD runs')
fig.savefig(path + 'sgd.png')

fig = plt.figure()
for i in range(400):
    cost = train_dropout(trX, trY)
    if i % 10 == 0:
        plt.plot(teX, predict(np.array(teX)), 'b-')
plt.plot(trX, trY, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('train with dropout, prediction using last 400 SGD runs, '
          'use one in ten')
fig.savefig(path + 'sgd_gap.png')

fig = plt.figure()
plt.plot(teX, predict(np.array(teX)), 'b-')
plt.plot(trX, trY, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('train with dropout, prediction using final weights')
fig.savefig(path + 'sgd_one.png')

plt.show()

