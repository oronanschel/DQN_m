import theano, lasagne
import theano.tensor as T
import math, csv, time, sys, os, pdb, copy
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
from learning import Adam, SGD, RMSProp, AdaDelta
from lasagne.layers import Conv2DLayer, cuda_convnet
import numpy as np

def get_activation(activation, m={"model_type":"linear"}):
  def conv_softmax(x):
    ax = 1
    e_x = T.exp(x - x.max(axis=ax, keepdims=True))
    return e_x / e_x.sum(axis=ax, keepdims=True)

  if activation == "softmax":
    if m["model_type"] == "conv":
      output = conv_softmax
    else:
      output = T.nnet.softmax
  elif activation == "tanh":
    output = T.tanh
  elif activation == "relu":
    output = T.nnet.relu
  elif activation == "linear":
    output = None
  elif activation == "sigmoid":
    output = T.nnet.sigmoid
  elif activation == "hard_sigmoid":
    output = T.nnet.hard_sigmoid
  return output

class AddBias():
  def __init__(self, init_val=None):
    self.b = init_val if init_val is not None else theano.shared(np.array([0.1], dtype="float32")[0])

  def get_output_for(self, inputs):
    return inputs+self.b

  def get_params(self):
    return [self.b]


def create_layer(inputs, model, dnn_type=True):
  if model["model_type"] == "conv":
    if dnn_type:
      import lasagne.layers.dnn as dnn
    conv_type = dnn.Conv2DDNNLayer if dnn_type else cuda_convnet.Conv2DCCLayer
    poolsize = tuple(model["pool"]) if "pool" in model else (1,1)
    stride = tuple(model["stride"]) if "stride" in model else (1,1)
    layer = conv_type(inputs, 
      model["out_size"], 
      filter_size=model["filter_size"], 
      stride=stride, 
      nonlinearity=get_activation(model["activation"], m=model),
      W=lasagne.init.HeUniform() if "W" not in model else model["W"],
      b=lasagne.init.Constant(.1) if "b" not in model else model["b"])

  elif model["model_type"] == "mlp" or model["model_type"] == "logistic":
    #layer = HiddenLayer(rng, input_dim, model["out_size"], activation=model["activation"])
    layer = lasagne.layers.DenseLayer(inputs, 
      num_units=model["out_size"],
      nonlinearity=get_activation(model["activation"]),
      W=lasagne.init.HeUniform() if "W" not in model else model["W"],
      b=lasagne.init.Constant(.1) if "b" not in model else model["b"])
  elif model["model_type"] == "logistic":
    raise NotImplementedError
    #layer = LogisticRegression(input_dim, model["out_size"], rng=rng, activation=model["activation"], 
    #  cost_type=model["loss_type"])
  elif model["model_type"] == "bias":
    layer = AddBias(model["b"] if "b" in model else None)
  elif model["model_type"] == "dropout":
    layer = lasagne.layers.DropoutLayer(inputs, p=0.5)
  elif model["model_type"] == "batchnorm":
    layer = lasagne.layers.batch_norm(inputs)
  return layer

class Model():
  def __init__(self, model, input_size=None, rng=1234, dnn_type=True):
    """
    example model:
    model = [{"model_type": "conv", "filter_size": [5,5], "pool": [1,1], "stride": [1,1], "out_size": 5},
             {"model_type": "conv", "filter_size": [7,7], "pool": [1,1], "stride": [1,1], "out_size": 15},
             {"model_type": "mlp", "out_size": 300, "activation": "tanh"},
             {"model_type": "classification", "out_size": 10, "activation": "tanh", "loss_type": "log_loss"}]
    """
    rng = np.random.RandomState(rng)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    lasagne.random.set_rng(rng) #set rng

    new_layer = tuple(input_size) if isinstance(input_size, list) else input_size
    self.model = model
    self.input_size = input_size
    self.dnn_type = dnn_type

    # create neural net layers
    self.params = []
    self.layers = []
    for i, m in enumerate(model):
      new_layer = create_layer(new_layer, m, dnn_type=dnn_type)
      if i < len(model)-1:
        if model[i+1]["model_type"] != "batchnorm":
          self.params += new_layer.get_params()
      else:
        self.params += new_layer.get_params()
      self.layers.append(new_layer)

  def apply(self, x, deterministic=False):
    last_layer_inputs = x
    last_model_type = None
    for i, m in enumerate(self.model):
      if last_model_type == "conv" and m["model_type"] != "conv":
        last_layer_inputs = last_layer_inputs.flatten(2)
      last_layer_inputs = self.layers[i].get_output_for(last_layer_inputs, deterministic=deterministic)
      last_model_type = m["model_type"]
    return last_layer_inputs

  def get_learning_method(self, l_method, **kwargs):
    if l_method == "adam":
      return Adam(**kwargs)
    elif l_method == "adadelta":
      return AdaDelta(**kwargs)
    elif l_method == "sgd":
      return SGD(**kwargs)
    elif l_method == "rmsprop":
      return RMSProp(**kwargs)

  def save_params(self):
    return [i.get_value() for i in self.params]

  def load_params(self, values):
    print "LOADING NNET..",
    for p, value in zip(self.params, values):
      p.set_value(value)
    print "LOADED"

  