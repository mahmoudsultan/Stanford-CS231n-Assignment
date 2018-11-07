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

  ###############################################################################
  # TODO(DONE): Compute the softmax loss and its gradient using explicit loops. #
  # Store the loss in loss and the gradient in dW. If you are not careful       #
  # here, it is easy to run into numeric instability. Don't forget the          #
  # regularization!                                                             #
  ###############################################################################
# compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    l_i = -correct_class_score + np.log(np.sum(np.exp(scores)))
    loss += l_i

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
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
  # compute scores matrix
  scores = X @ W

  # get the correct class scores indicies
  train_count = X.shape[0]
  correct_classes_list = y.flatten().tolist()
  correct_scores_indices = range(0, train_count), correct_classes_list
  
  # get neg. correct class score for each example
  correct_scores_neg = scores[correct_scores_indices] * -1

  # compute the log part
  sum_logs = np.log(np.sum(np.exp(scores), axis=1))

  losses_vector = correct_scores_neg + sum_logs
  loss = (np.sum(losses_vector) / train_count) + (reg * np.sum(W * W))
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

