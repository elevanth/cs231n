import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  tmp = 0.0
  for i in xrange(num_train):
    score = X[i].dot(W)
    e_score = np.exp(score)
    sum_score = np.sum(e_score)
    for j in xrange(num_class):
      # tmp += exp(score[j])
      if j == y[i]:
        dW[:,j] -= X[i]
      dW[:,j] += e_score[j] * X[i] / sum_score
    loss += np.log(sum_score) - score[y[i]]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  scores = X.dot(W)
  e_scores = np.exp(scores)
  sum_scores = np.sum(e_scores, axis=1)
  loss = np.log(sum_scores) - scores[xrange(num_train), y]
  loss = np.sum(loss)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  tmp = e_scores / sum_scores.reshape(num_train, -1)
  tmp[xrange(num_train), y] -= 1
  dW = X.T.dot(tmp) / num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

