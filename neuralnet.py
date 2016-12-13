import numpy as np
from scipy.special import expit
import sys
import os

class NeuralNetwork(object):
    """ Feedforward neural network with a single hidden layer
        Params:
        n_output: int: number of output units, equal to num class labels
        n_features: int: number of features in the input dataset
        n_hidden: int: (default 30): num hidden units
        l1: float (default: 0.0) - lambda value for L1 regularization
        l2: float(default: 0.0) - lambda value for L2 regularization
        epochs: int (default = 500) - passes over training set
        eta: float (default: 0.001) - learning reate
        alpha: float (default: 0.0) - momentum constant - multiplied with gradient of previous pass through set
        decrease_const: float (default 0.0) - shrinks learning rate after each epoch: eta = eta / (1 + epoch*decrease_const)
        shuffle: bool (default: True) - shuffles training data each pass to prevent circles
        minibatches: int (default: 1) - divides training data into batches for efficiency
        random_state: int (default: None) - sets random state for initializing weights
    """

    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                    alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        # number of output neurons
        self.n_output = n_output
        # number of input features each training sample has
        self.n_features = n_features
        # number of hidden neurons
        self.n_hidden = n_hidden
        # w1: weights for input -> hidden
        # w2: weights for hidden -> output
        self.w1, self.w2 = self._initialize_weights()
        # L1 and L2 regularization consts
        self.l1 = l1
        self.l2 = l2
        # number of passes over training set
        self.epochs = epochs
        # learning rates
        self.eta = eta
        self.alpha = alpha
        # how much to decrease the learning rate by
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _initialize_weights(self):
        """ init weights with random nums
        """
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features
                               + 1).reshape(self.n_hidden, self.n_features))
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden+1)
                               .reshape(self.n_output, self.n_hidden + 1))
        return w1, w2

    def _encode_labels(self, y, k):
        """ Encode labels into a one-hot representation
            Params:
            y: array of num_samples, contains the target class labels for each training example.
            For example, y = [2, 1, 3, 3] -> 4 training samples, and the ith sample has label y[i]
            k: number of output labels
            returns: onehot, a matrix of labels by samples. For each column, the ith index will be
            "hot", or 1, to represent that index being the label.
        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0

        return onehot

    def _sigmoid(self, z):
        """ Compute the logistic function.
        """
        #use scipy.special.expit
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute the gradient of the logistic function
        """
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, column=True):
        if(column):
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        else:
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X

        return X_new

    def _feedforward(self, X, w1, w2):
        """ Compute feedforward step
            Params:
            X: matrix of num_samples by num_features, input layer with samples and features
            w1: matrix of weights from input layer to hidden layer. Dimensionality of num_hidden_units by num_features + 1 (bias)
            w2: matrix of weights from hidden layer to output layer. Dimensionality of num_output_units (equal to num class labels) by num_hidden units + 1 (bias)
        """
        #the activation of the input layer is simply the input matrix plus bias unit, added for each sample.
        a1 = self._add_bias_unit(X)
        #the input of the hidden layer is obtained by applying our weights to our inputs. We essentially take a linear combination of our inputs
        z2 = w1.dot(a1.T)
        #applies the logistic function to obtain the input mapped to a distrubution of values between 0 and 1
        a2 = self._sigmoid(z2)
        #add a bias unit to activation of the hidden layer.
        a2 = self._add_bias_unit(a2, column=False)
        #compute input of output layer in exactly the same manner. This can be generalized to many layers!
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)

        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2 regularization cost
        """
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) +
                                np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        """Compute L1-regularization cost"""
        return (lambda_/2.0) * (np.abs(w1[:, 1:]).sum() +
                                np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """ Compute the cost function.
            Params:
            y_enc: array of num_labels x num_samples. class labels one-hot encoded
            output: matrix of output_units x samples - activation of output layer from feedforward
            w1: weight matrix of input to hidden layer
            w2: weight matrix of hidden to output layer
            """
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Computes the gradient using backpropagation
            Params:
            a1: array of n_samples by features+1 - activation of input layer (just input plus bias)
            a2: activation of hidden layer
            a3: activation of output layer
            z2: input of hidden layer
            y_enc: onehot encoded class labels
            w1: weight matrix of input layer to hidden layer
            w2: weight matrix of hidden to output layer
            returns: gradient of weight matrix w1, gradient of weight matrix w2
        """
        #backpropagate our error
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, column=False)
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        #get rid of the bias row
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
         #regularization
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):
        if len(X.shape)!=2:
            raise AttributeError("something got messed up")

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        #z3 is of dimension output units x num_samples. each row is an array representing the likelihood that the sample belongs to the class label given by the index...
        #ex: first row of z3 = [0.98, 0.78, 0.36]. This means our network has 3 output units = 3 class labels. And this instance most likely belongs to the class given by the label 0.
        y_pred = np.argmax(a3, axis = 0)
        self.predicted_labels = y_pred
        return y_pred


    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data
            Params:
            X: matrix of samples x features. Input layer
            y: target class labels of the training instances (ex: y = [1, 3, 4, 4, 3])
        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        #pass through the dataset
        for i in range(self.epochs):
            self.eta /= (1 + self.decrease_const*i)
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                #feed feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)

                #compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha*delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha*delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self

if __name__ == '__main__':
    n = NeuralNetwork(5,2)
    print len(n.w1)
    labels = [1, 2, 2, 0, 29]
    y_enc = n._encode_labels(np.asarray(labels), 30)
    print y_enc
