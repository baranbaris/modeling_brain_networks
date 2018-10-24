import numpy as np

# forward and backward pass
# called for faster calculation since the intermediate operations are omitted
# such as loss calculation
def for_and_back(x, w, lambda_, epoch):

    o = np.matmul(x, w.T)
    l1 = x - o
    # partial derivative of the loss (i.e. ridge regression) with respect to w
    dw = np.matmul(2*l1.T, -x) + 2*lambda_*w

    return dw

# forward and backward pass with calculation of loss

def for_and_back_debug(x, w, lambda_, epoch):

    o = np.matmul(x, w.T)
    l1 = x - o
    # calculate l2 loss
    l2 = np.multiply(l1, l1)
    l2 = np.sum(l2, axis=0)/x.shape[0]
    # calculate l2 regularization
    regs = np.multiply(w, w)
    regs = lambda_ *np.sum(regs, axis=0)
    # loss : ridge regression, l2 loss  + l2 regularization
    loss = l2 + regs
    # partial derivative of the loss (i.e. ridge regression) with respect to w
    dw = np.matmul(2*l1.T, -x) + 2*lambda_*w
    # print epoch loss
    print('epoch : %.4d \t loss %.10f' % (epoch+1, np.sum(loss)))
    return dw


# optimization for DIRECTED brain networks

# x is the matrix of windowed data (i.e. batch) with the dimensions (W,N)
# where W denotes the lenght of window
# and N denotes the number of brain regions which is the number of nodes
# in the resulting graph

# epochs : number of epochs (int)

# alpha : learning rate

# lambda_ : regularization parameter

# debug_choice : choose 0 if you want to observe the loss values
# and its change during training, select 0 if you want a process without
# any prompts of the process or a faster training
def minimize_dir(x, epochs, alpha, lambda_, debug_choice=0):

    debug_options = ['run', 'debug']

    funct_dict = {
        debug_options[0] : for_and_back,
        debug_options[1] : for_and_back_debug
    }


    nodes = x.shape[1]
    # seed for random initialization of the weight matrix
    np.random.seed(42)
    weights = np.random.rand(nodes, nodes)

    for epoch in range(epochs):
        # self connections are omitted
        np.fill_diagonal(weights, 0)
        # get the derivative
        dw = funct_dict[debug_options[debug_choice]](x, weights, lambda_, epoch)
        # update weights
        weights = weights - alpha*dw

    # again self connections are omitted
    np.fill_diagonal(weights, 0)
    return weights.ravel()

# optimization for UNDIRECTED brain networks

# meaning of the parameters is the same as the above function (minimize_dir)

def minimize_und(x, epochs, alpha, lambda_, debug_choice=0):

    debug_options = ['run', 'debug']

    funct_dict = {
        debug_options[0] : for_and_back,
        debug_options[1] : for_and_back_debug
    }


    nodes = x.shape[1]
    np.random.seed(42)
    weights = np.random.rand(nodes, nodes)
    weights = (weights + weights.T)/2

    for epoch in range(epochs):
        np.fill_diagonal(weights, 0)
        # get the derivative
        dw = funct_dict[debug_options[debug_choice]](x, weights, lambda_, epoch)
        # weight sharing, to make the weight matrix symmetric hence undirected
        dw = (dw + dw.T)/2
        weights = weights - alpha*dw

    # again self connections are omitted
    np.fill_diagonal(weights, 0)
    return weights.ravel()
