# README #

This is a program that learns to play Atari video games from pixel information alone, using RL and convnets.

# Installing
Clone all repos mentioned below into the same parent directory.

Clone this repo (replace 'username' with your bitbucket username):
```git clone https://username@bitbucket.org/rllabmcgill/atari_release.git```

Clone ALE: 

```git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git```

Install ALE:

Use the manual found at:

```https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/doc/manual/manual.pdf```

Or these are the current instructions (Ubuntu)

```sudo apt-get install cmake```

```sudo apt-get install libsdl1.2-dev```

```cd Arcade-Learning-Environment```

```cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON```

```make -j 4```

Add Arcade-Learning-Environment folder to your PYTHONPATH 
(eg: on Ubuntu add the following line at the end of your .bashrc file in your home directory)
```export PYTHONPATH=${PYTHONPATH}:/home/user/path/to/Arcade-Learning-Environment```

Download the aleroms repo and keep in parent of this repo.
```
git clone https://username@bitbucket.org/rllabmcgill/aleroms.git
```

Install Requirements

Python-Dev : 		```sudo apt-get install python-dev```

Scipy Reqs : 		```sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran```

Numpy : 		```sudo apt-get install python-numpy```

theano : 		```sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip```

Lasagne : 		```sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip```

cv2 : 			```sudo apt-get install python-opencv```

matplot-lib : 		```sudo apt-get install python-matplotlib```

pylearn2:		
				```git clone git://github.com/lisa-lab/pylearn2.git``` 
				```python setup.py develop```

IMPORTANT: you must set theano.config.floatX to float32.
You can do this in 2 ways:

Edit your ~/.theanorc file to have 

```
[global]
floatX = float32
```

OR add a theano flag before running training the network. e.g. ```THEANO_FLAGS=floatX=float32 python train_q.py --rom pong```


# Training

```
python train_q.py --rom "game_name"
```

The trained model will be saved in directory under models/. By default, the directory name will be a timestamp of the launch time. To choose the directory name, add the flag --folder-name and the wanted name. e.g. ```python train_q.py --rom breakout --folder-name breakout_dqn```. This will save the model under models/breakout_dqn.

Currently, DQN and double DQN (vanilla and tuned) are available.

For more flags, just run ```python train_q.py --help```

NOTE: To speed up training, running it on GPU makes a huge difference. 

Furthermore, having CUDNN v3 or v4 also improves speed by around 15%. 
(CUDNN can be downloaded at https://developer.nvidia.com/cudnn). 

Finally, setting the theano flag allow_gc to false can also help speed up training.


# Testing
To view trained model:
```
python run_best_model.py "model_path"
```
e.g. ```python run_best_model.py models/breakout_dqn/last_model.pkl```