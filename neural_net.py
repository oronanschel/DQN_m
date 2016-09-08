import theano, sys
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from nnet import Model


class DeepQNetwork():
  def __init__(self, model_network=None,gamma=0.99, learning_method="rmsprop", 
    batch_size=32, input_size=None, learning_params=None, dnn_type=True, clip_delta=0, 
    scale=255., double_q=False, prioritized_exp_replay=False):
    x = T.ftensor4()
    next_x = T.ftensor4()
    a = T.ivector()
    r = T.fvector()
    terminal = T.ivector()

    self.x_shared = theano.shared(np.zeros(tuple([batch_size]+input_size[1:]), dtype='float32'))
    self.next_x_shared = theano.shared(np.zeros(tuple([batch_size]+input_size[1:]), dtype='float32'))
    self.a_shared = theano.shared(np.zeros((batch_size), dtype='int32'))
    self.terminal_shared = theano.shared(np.zeros((batch_size), dtype='int32'))
    self.r_shared = theano.shared(np.zeros((batch_size), dtype='float32'))

    
    self.Q_model = Model(model_network, input_size=input_size, dnn_type=dnn_type)
    self.Q_prime_model = Model(model_network, input_size=input_size, dnn_type=dnn_type)
    
    if double_q:
      alt_actions = T.argmax(self.Q_model.apply(next_x/scale), axis=1)
      alt_actions = theano.gradient.disconnected_grad(alt_actions)
      y = r + (T.ones_like(terminal)-terminal)*gamma*\
      self.Q_prime_model.apply(next_x/scale)[T.arange(alt_actions.shape[0]), alt_actions]
    else:
      y = r + (T.ones_like(terminal)-terminal)*gamma*T.max(self.Q_prime_model.apply(next_x/scale), axis=1)
    
    all_q_vals = self.Q_model.apply(x/scale)
    q_vals = all_q_vals[T.arange(a.shape[0]), a]

    td_errors = y-q_vals

    """
    if clip_delta > 0:
      td_errors = td_errors.clip(-clip_delta, clip_delta)
    cost = 0.5*td_errors**2
    """
    if clip_delta > 0:
      #TOOK THIS FROM GITHUB CODE

      # If we simply take the squared clipped diff as our loss,
      # then the gradient will be zero whenever the diff exceeds
      # the clip bounds. To avoid this, we extend the loss
      # linearly past the clip point to keep the gradient constant
      # in that regime.
      # 
      # This is equivalent to declaring d loss/d q_vals to be
      # equal to the clipped diff, then backpropagating from
      # there, which is what the DeepMind implementation does.
      quadratic_part = T.minimum(abs(td_errors), clip_delta)
      linear_part = abs(td_errors) - quadratic_part
      cost = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
    else:
      cost = 0.5 * td_errors ** 2
    #"""

    cost = T.sum(cost)

    print self.Q_model.params
    self.learning_method = self.Q_model.get_learning_method(learning_method, **learning_params)
    grads = T.grad(cost, self.Q_model.params)
    param_updates = self.learning_method.apply(self.Q_model.params, grads)

    target_updates = OrderedDict()
    for t, b in zip(self.Q_prime_model.params, self.Q_model.params):
      target_updates[t] = b

    givens = {x:self.x_shared, a:self.a_shared, r:self.r_shared, 
    terminal:self.terminal_shared, next_x:self.next_x_shared}

    print "building"
    self.train_model = theano.function([], td_errors, updates=param_updates, givens=givens)
    print "compiled train_model (1/3)"
    self.pred_score = theano.function([], all_q_vals, givens={x:self.x_shared})
    print "compiled pred_score (2/3)"
    self.update_target_params = theano.function([], [], updates=target_updates)
    print "compiled update_target_params (3/3)"
    self.update_target_params()
    print "updated target params"

  def predict_move(self, x):
    self.x_shared.set_value(x)
    return np.argmax(self.pred_score(), axis=1)

  def get_q_vals(self, x):
    self.x_shared.set_value(x)
    return self.pred_score()

  def predict_score_with_move(self, x, a):
    self.x_shared.set_value(x)
    return self.pred_score()[a]

  def train_conv_net(self, train_set_x, next_x, actions, r, terminal):
    self.x_shared.set_value(train_set_x)
    self.next_x_shared.set_value(next_x)
    self.a_shared.set_value(actions)
    self.r_shared.set_value(r)
    self.terminal_shared.set_value(terminal)
    return self.train_model()

  def save_params(self):
    return self.Q_model.save_params()

  def load_params(self, values):
    self.Q_model.load_params(values)



