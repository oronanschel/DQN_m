import sys, pdb, time, random, os, cv2, datetime, csv, glob, theano
import numpy as np
from random import randrange
#from ale_python_interface import ALEInterface
from ALE.ale_python_interface.ale_python_interface import ALEInterface
import matplotlib
from pygame.examples.testsprite import screen_dims
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import pickle as pkl
import theano.tensor as T
import scipy
from theano import ProfileMode
from neural_net import DeepQNetwork
from exp_replay import DataSet
from plot_learning import plot

sys.setrecursionlimit(50000)


def load_params(model_path):
  mydir = "/".join(model_path.split("/")[:-1])
  model_params = pkl.load(open(os.path.join(mydir, 'model_params.pkl'), 'rb'))
  return model_params

def create_dir(p):
  try:
    os.makedirs(p)
  except OSError, e:
    if e.errno != 17:
      raise # This was not a "directory exist" error..

def filecreation(model_params, folder_name=None):
  tempdir = os.path.join(os.getcwd(), "models")
  create_dir(tempdir)
  folder_name = folder_name if folder_name is not None else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  mydir = os.path.join(tempdir, folder_name)
  create_dir(mydir)
  pkl.dump(model_params, open(os.path.join(mydir, 'model_params.pkl'), "wb"))
  return mydir

class Trainer(object):
  def create_results_file(self):
    self.prog_file = os.path.join(self.mydir, 'training_progress.csv')
    data_file = open(self.prog_file, 'wb')
    data_file.write('epoch,mean_score,mean_q_val\n')
    data_file.close()

  def update_results(self, epoch, ave_reward, ave_q):
    # if it isn't, then we are testing and watching a game.
    # no need to update a file.
    if self.params.nn_file is None:
      fd = open(self.prog_file,'a')
      fd.write('%d,%f,%f\n' % (epoch, ave_reward, ave_q))
      fd.close()
      plot(self.mydir)

  def __init__(self, model_params, ale_env, folder_name):
    self.init_time = time.time()
    # nn_file only present when watching test
    if model_params.nn_file is None:
      self.mydir = filecreation(model_params, folder_name)
      self.create_results_file()
      self.params = model_params
    else:
      old_params = load_params(model_params.nn_file)

      self.params = model_params
      self.params.rom_path = old_params.rom_path

    #ale_env.setInt('frame_skip', self.params.frame_skip)
    ale_env.setFloat('repeat_action_probability', 0.)
    ale_env.setBool('color_averaging', True)
    ale_env.loadROM(self.params.rom_path)


    #chooses which convnet to use based on cudnn availability
    self.params.USE_DNN_TYPE = False
    if theano.config.device.startswith("gpu"):
      self.params.USE_DNN_TYPE=theano.sandbox.cuda.dnn.dnn_available()
    if self.params.USE_DNN_TYPE:
      print "USING CUDNN"
    else:
      print "WARNING: NOT USING CUDNN. TRAINING WILL BE SLOWER."


    self.rng = np.random.RandomState(1234)
    self.noop_action = 0 

    self.USE_MAX_SCREEN_LAST_TWO = False #True

    self.frame_count = 0.
    self.best_reward = -100.
    self.max_frames_per_game = 18000

    self.ale = ale_env
    self.legal_actions = self.ale.getMinimalActionSet()
    

    self.screen_dims = self.ale.getScreenDims()

  def cap_reward(self, reward, testing=False):
    if self.params.do_cap_reward and not testing:
      if reward > 0:
        score = 1
      elif reward < 0:
        score = -1
      else:
        score = 0
      return score
    else:
      return reward

  def _init_ep(self):
    num_actions = np.random.randint(4, self.params.max_start_nullops)
    x = []
    self.last_screen = np.zeros((210, 160), dtype='uint8')
    for i in range(num_actions):
     
      self.ale.act(self.noop_action)
      if i >= num_actions-self.params.phi_length:
        x.append(self.get_observation())
    return x

  def act(self, action, testing=False):
    reward = 0
    for i in range(self.params.frame_skip):
      reward += self.ale.act(self.legal_actions[action])
    x = self.get_observation()
    return self.cap_reward(reward, testing), self.cap_reward(reward, True), x

  def get_observation(self):
    screen = self.ale.getScreenGrayscale().reshape(self.screen_dims[1], self.screen_dims[0])
    if self.params.resize_method == "crop":
      resized = scipy.misc.imresize(screen, size=(110,84))[self.params.offset:self.params.offset+84, :]
    elif self.params.resize_method == "scale":
      resized = cv2.resize(screen, (84, 84), interpolation=cv2.INTER_LINEAR)
    else:
      print "wrong resize_method, only have crop and scale"
      raise NotImplementedError
    return resized

  def get_action(self, epsilon, x):
    if np.random.rand() < epsilon:
      return np.random.randint(len(self.legal_actions))
    else:
      return self.model.predict_move(np.array(x).reshape(1,4,84,84))

  def save_model(self, total_reward, skip_best=False):
    if total_reward >= self.best_reward and not skip_best:
      self.best_reward = total_reward
      pkl.dump(self.model.save_params(), open(os.path.join(self.mydir, 'best_model.pkl'), "w"), protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(self.model.save_params(), open(os.path.join(self.mydir, 'last_model.pkl'), "w"), protocol=pkl.HIGHEST_PROTOCOL)
    print "Saved model"

  def run_training_episode(self):
    raise NotImplementedError

  def get_learning_params(self):
    d = {}
    if self.params.update_rule == "rmsprop":
      d["lr"] = self.params.learning_rate
      d["eps"] = self.params.rms_epsilon
      d["rho"] = self.params.rms_decay
    elif self.params.update_rule == "adam":
      d["lr"] = self.params.learning_rate
    return d

  def get_epsilon(self):
    #linear descent from 1 to 0.1 starting at the replay_start_time
    replay_start_time = max([self.frame_count-self.params.replay_start_size, 0])
    epsilon = self.params.epsilon_start
    epsilon -= (self.params.epsilon_start - self.params.epsilon_min)*\
      (min(replay_start_time, self.params.epsilon_decay)/self.params.epsilon_decay)
    return epsilon

  def get_mean_q_val(self, batch=1000):
    imgs = self.exp_replay.random_batch(batch, random_selection=True)
    return np.mean(np.max(self.model.get_q_vals(imgs[0]),axis=1))

  def run_testing(self, epoch):
    total_reward = 0
    num_games = 0
    original_frame_count = self.frame_count
    rem = self.params.steps_per_test
    while(self.frame_count - original_frame_count < self.params.steps_per_test):
      reward, fps = self.run_training_episode(self.max_frames_per_game, testing=True)
      print ("TESTING: %d fps,\t" % fps),
      print ("%d frames,\t" % self.ale.getEpisodeFrameNumber()),
      self.ale.reset_game()
      print "%d points,\t" % reward,
      rem = self.params.steps_per_test-(self.frame_count - original_frame_count) 
      print "rem:", rem,
      print "ETA: %d:%02d" % (max(0, rem/60/fps*4), ((rem/fps*4)%60) if rem > 0 else 0)
      total_reward += reward
      num_games += 1
    self.frame_count = original_frame_count
    mean_reward = round(float(total_reward)/num_games, 2)
    print "AVERAGE_SCORE:", mean_reward
    if type(self) is Q_Learning:
      mean_q = self.get_mean_q_val() if self.params.nn_file is None else 1
    else:
      mean_q = 1
    self.update_results(epoch+1, mean_reward, mean_q)

  def train(self):
    cumulative_reward = 0
    counter = 0
    for i in range(self.params.epochs):
      start_frames = self.frame_count
      frames_rem = self.params.steps_per_epoch
      while self.frame_count-start_frames < self.params.steps_per_epoch:
        total_reward, fps = self.run_training_episode(self.max_frames_per_game)
        cumulative_reward += total_reward
        frames_rem = self.params.steps_per_epoch-(self.frame_count-start_frames)
        print ("ep %d,\t") % (counter+1),
        print ("%d fps,\t" % fps), 
        print ("%d frames,\t" % self.ale.getEpisodeFrameNumber()),
        self.ale.reset_game()
        print ('%d points,\t' % total_reward),
        print ('%.1f avg,\t' % (float(cumulative_reward)/(counter+1))), 
        print "%d rem," % frames_rem, 'eps: %.4f' % self.get_epsilon(),
        print "ETA: %d:%02d" % (max(0, frames_rem/60/fps*4), ((frames_rem/fps*4)%60) if frames_rem > 0 else 0)
        counter +=1
      
      if self.params.nn_file is None:
        self.save_model(total_reward)

      self.run_testing(i)

class DQN_Trainer(Trainer):
  def __init__(self, **kwargs):
    super(DQN_Trainer, self).__init__(**kwargs)

  def run_training_episode(self, max_steps, testing=False):
    def get_new_frame(new_frame, x):
      new_x = np.empty((4, 84, 84), dtype="float32")
      new_x[0:3] = x[-3:]
      new_x[-1] = new_frame
      return new_x

    start_time = time.time()

    total_reward = 0
    data_set = self.test_replay if testing else self.exp_replay
    start_frame_count = self.frame_count
    x = self._init_ep()
    game_over = self.ale.game_over()
    num_lives = self.ale.lives()
    while not game_over:
      self.frame_count += 1
      epsilon = self.get_epsilon() if not testing else self.params.optimal_eps
      chosen_action = self.get_action(epsilon, x)
      reward, raw_reward, new_frame = self.act(chosen_action, testing=testing)

      game_over = self.ale.game_over() or (self.frame_count-start_frame_count) > max_steps
      new_num_lives = self.ale.lives()
      life_death = (new_num_lives < num_lives and not testing and self.params.death_ends_episode)
      num_lives = new_num_lives
      
      data_set.add_sample(x[-1], chosen_action, reward, game_over or life_death)
      x = get_new_frame(new_frame, x)
      total_reward += raw_reward
      if self.frame_count > self.params.replay_start_size and not testing:
        if self.frame_count % self.params.update_frequency == 0:
          self.learn()
        if self.frame_count % self.params.freeze_interval == 0:
          print "updated weights"
          self.model.update_target_params()
    #print self.get_mean_q_val(50)
    fps = round((self.frame_count - start_frame_count)/(time.time()-start_time), 2)
    fps = self.ale.getEpisodeFrameNumber()/(time.time()-start_time)
    return total_reward, fps


  def learn(self):
    x, a, r, next_x, term = self.exp_replay.random_batch(self.params.batch_size)
    td_errors = self.model.train_conv_net(x, next_x, a, r, term)
    return td_errors

class Q_Learning(DQN_Trainer):
  def __init__(self, **kwargs):
    super(Q_Learning, self).__init__(**kwargs)
    model_network = [{"model_type": "conv", "filter_size": [8,8], "pool": [1,1], "stride": [4,4],
                     "out_size": 32, "activation": "relu"},
                     {"model_type": "conv", "filter_size": [4,4], "pool": [1,1], "stride": [2,2],
                     "out_size": 64, "activation": "relu"},
                     {"model_type": "conv", "filter_size": [3,3], "pool": [1,1], "stride": [1,1],
                     "out_size": 64, "activation": "relu"},
                     {"model_type": "mlp", "out_size": 512, "activation": "relu"},
                     {"model_type": "mlp", "out_size": len(self.legal_actions), "activation": "linear"}]

    #if self.params.double_q:
    #  model_network[-1]["b"] = None
    #  model_network.append({"model_type": "bias"})
    learning_params = self.get_learning_params()
    self.model = DeepQNetwork(model_network=model_network, double_q=self.params.double_q,
      learning_method=self.params.update_rule, dnn_type=self.params.USE_DNN_TYPE, clip_delta=self.params.clip_delta,
      input_size=[None,4,84,84], batch_size=self.params.batch_size, learning_params=learning_params, 
      gamma=self.params.discount)
    if self.params.nn_file is not None:
      self.model.load_params(pkl.load(open(self.params.nn_file, 'r')))

    self.exp_replay = DataSet(84, 84, self.rng, max_steps=self.params.replay_memory_size, 
      phi_length=4)
    self.test_replay = DataSet(84, 84, self.rng, max_steps=4, phi_length=4)


models={1:Q_Learning}

def get_model(ind):
  return models[ind]


if __name__ == "__main__":
  pass
