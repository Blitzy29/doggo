# DOGGO


Reinforcement Learning to make a Doggo immortal.


# Table of contents

* [What is a Doggo?](#what-is-a-doggo?)
* [Reinforcement Learning](#reinforcement-learning)
* [Getting Started](#getting-started)
* [How to use it (simplified)](#how-to-use-it-(simplified))
* [Project organisation](#project-organisation)
<!--* [License](#license)-->
<!--* [Known bugs](#known-bugs)-->

## [What is a Doggo?](src/data/README.md)

Remember tamagotchi? Doggo are similar. It is a digital pet, which one has to take care of by taking some actions, otherwise the Doggo will die.

A Doggo is represented by 3 characteristics:

* Food
* Fat
* Affection

All of these characteristics are between 0 and 1 and help to determine the happiness of the Doggo:

<p align="center">
<img src="reports/figures/README/doggo_happiness.png" width="350">
</p>

Once the happiness reaches 0, the Doggo dies.

Food and affection will diminish in time. In my test, it decreases by a value of 0.01 every minute (or step). Fat doesn't change with time. In order to let the doggo live, actions can be taken to improve food and affection, or reduce fat.: feeding, playing or walking

The idea and creation of the Doggo digital pet is a [project]() by [Romaric Thiam](https://github.com/RTH00).


## [Reinforcement Learning](src/models/README.md)

The primary goal of this project is to be able to make Doggos immortal. We want to find a modelthat will take the correct action to ensure that the Doggo will continue to live forever, with no restrictions on the number of actions.

(Note: later, I want to optimize the number of actions taken by the model, and optimize a long period without actions).

#### State

A state is define as the combination of:

* the 3 characteristics (food, fat, affection)
* a boolean, whether an action can be taken

#### Action

3 actions are possible: feeding, playing and walking. Feeding will increase food and fat ; playing and walking will increase affection and decrease fat.

#### Reward

As a reward, we conside the happiness of the new state (therefore between 0 and 1). A (negative) reward of $-10$  is applied of the Doggo dies.


### Implemented technics

* [Monte Carlo](src/models/value_based/monte_carlo/README.md)
* [SARSA](src/models/value_based/sarsa/README.md)
* [Q-Learning](src/models/value_based/qlearning/README.md)
* [Double Q-Learning](src/models/value_based/doubleqlearning/README.md)

More information can be found to the specific documentation for each method, as well as how to use it.

## Getting started

Requirements

* Python 3.7

In a local environment, run

~~~
pip3 install -r requirements.txt
~~~


## How to use it (simplified)

### [Launch a Monte-Carlo test](src/models/value_based/monte_carlo/README.md)

~~~python
train_model_MonteCarlo.py [-h] [--render_episode [RENDER_EPISODE]]
                            [--run_episode RUN_EPISODE]
                            [--update_episode_division {0,1,5,10,20,50}]
                            [--play_music [PLAY_MUSIC]]
                            method_id
~~~

### [Launch a SARSA test](src/models/value_based/sarsa/README.md)

~~~python
train_model_SARSA.py [-h] [--render_episode [RENDER_EPISODE]]
	                            [--run_episode RUN_EPISODE]
	                            [--update_episode_division {0,1,5,10,20,50}]
	                            [--play_music [PLAY_MUSIC]]
	                            method_id
~~~


### [Launch a Q-Learning test](src/models/value_based/qlearning/README.md)

~~~python
train_model_QLearning.py [-h] [--render_episode [RENDER_EPISODE]]
	                            [--run_episode RUN_EPISODE]
	                            [--update_episode_division {0,1,5,10,20,50}]
	                            [--play_music [PLAY_MUSIC]]
	                            method_id
~~~


### [Launch a Double Q-Learning test](src/models/value_based/doubleqlearning/README.md)

~~~python
train_model_DoubleQLearning.py [-h] [--render_episode [RENDER_EPISODE]]
			                            [--run_episode RUN_EPISODE]
			                            [--update_episode_division {0,1,5,10,20,50}]
			                            [--play_music [PLAY_MUSIC]]
			                            method_id
~~~

### [Visualize current process](src/visualization/README.md)

~~~
python3 src/visualization/app_currently_running.py
~~~
~~~
http://127.0.0.1:8052/
~~~

<p align="center">
<img src="reports/figures/README/Doggo - Currently Running - Part 1.png" width="500">
</p>


### [Analysis](src/visualization/README.md)

~~~
python3 src/visualization/app_analysis.py
~~~
~~~
http://127.0.0.1:8051/
~~~

<p align="center">
<img src="reports/figures/README/Doggo - Analysis - Part 1.png" width="500">
</p>


## Project Organization

    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── figures        <- Generated graphics and figures
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── log            <- Logs using a default name: 'YYYYmmdd_HHMMSS_file_handler_name'
    │   │   └── pid        <- Current processes, defined by a pid
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── dog_configuration <- json dog configuration
    │   │   └── env_dog.py <- dog environment
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── value_based
    │   │   │   └── sarsa
    │   │   │       ├── config <- configurations for the tests
    │   │   │       ├── rl_tools_sarsa.py <- RL tools specific to SARSA
    │   │   │       └── train_model_SARSA.py <- SARSA training
    │   │   │
    │   │   ├── run_comparison.py <- run comparison of tests in configuration
    │   │   ├── run_evolution_episode.py <- plot the result of a single episode
    │   │   └── run_one_episode.py <- run a full-exploitation episode
    │   │
    │   ├── utils          <- utilities, functions which can safely be used in all programs
    │   │   ├── formats.py
    │   │   ├── logger_module.py
    │   │   ├── maths.py
    │   │   ├── music.py
    │   │   ├── plots.py
    │   │   └── utils_dict.py
    │   │
    │   └── visualization  <- Scripts/Apps to create exploratory and results oriented visualizations
    │       ├── app_analysis.py <- run a Dash dashboard for analysis
    │       ├── app_currently_running.py <- run a Dash dashboard for observing current processes
    │       ├── plot_happiness.py <- tools for plot for happiness
    │       ├── rl_plots_comparison.py <- tools for plot for comparison
    │       └── rl_plots_evolution.py <- tools for plot for evolution
    │
    ├── LICENSE.md         <- License
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

<!--## Known Bugs-->

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
