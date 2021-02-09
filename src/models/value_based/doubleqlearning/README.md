# Double Q-Learning

## Launch a test

~~~python
train_model_DoubleQLearning.py [-h] [--render_episode [RENDER_EPISODE]]
                            [--run_episode RUN_EPISODE]
                            [--update_episode_division {0,1,5,10,20,50}]
                            [--play_music [PLAY_MUSIC]]
                            method_id
~~~

* render\_episode: render information about every episode (in log)
* [run\_episode: run an full-exploitation episode every X percent]
* update\_episode\_division: save Q and plot intermediate results every X percent
* play the spam song when finished
* method\_id: name of the config file under src/models/value\_based/sarsa/config

~~~python
python -m src.models.value_based.doubleqlearning.train_model_DoubleQLearning doubleqlearning_test --render_episode --run_episode 10 --update_episode_division 10 --play_music
~~~


### Configuration file

~~~json
{
  "method": "Double_Q_Learning"
  , "epsilon": {
    "init_epsilon": 1
    , "decay_epsilon": "per_state"
    , "min_epsilon": 0
  }
  , "alpha": {
    "init_alpha": 0.5
    , "decay_alpha": "fixed"
    , "min_alpha": 0.5
  }
  , "init_Q_type": "ones"
  , "n_episodes": 10000
  , "nmax_steps": 1440
  , "gamma": 0.95
  , "start_at_random": false
  , "KLdiv_convergence": 1e-10
}
~~~

* method: name of the method
* epsilon:
	* init\_epsilon: initialisation value of epsilon
	* decay\_epsilon: "fixed", "per\_state", "per\_episode"
	* min\_epsilon: minimum epsilon possible. Careful when used with fixed, it still has an impact
* alpha
	* init\_alpha: initialisation value of alpha (between 0 and 1)
	* decay\_alpha: "fixed", "per\_state", "per\_episode"
	* min\_alpha: minimum alpha possible. Careful when used with fixed, it still has an impact
* init\_Q\_type: all Q-values are initiated with:
	* if "ones": 1
	* if "zeros": 0
	* if "random": random values between 0 and 1
	* if "optimum": the maximum reward possible over the episode
* n\_episodes: number of episodes to be run in this test
* nmax\_steps: number of steps per episode to be run in this test
* gamma: gamma-value
* start\_at\_random: 
	* if False, every episode starts with 0.50 of food, 0.00 of fat, 0.50 of affection, and action cane be taken (this is similar to the real initial state of a Doggo and is always used in full-exploitation episode)
	* if True, every episode starts with random values for the 3 characteristics, and action can be taken 
* [KLdiv\_convergence: non-implemented yet]


--------
