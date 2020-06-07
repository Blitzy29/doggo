# DOGGO


Reinforcement Learning to make a Doggo immortal.


# Table of contents

* [What is a Doggo?](#what-is-a-doggo?)
* [Reinforcement Learning](#reinforcement-learning)


## What is a Doggo?

Doggo is first a project by [Romaric Thiam](https://github.com/RTH00). Remember tamagotchi? Doggo are similar. It is a digital pet, which one has to take care of by taking some actions, otherwise the Doggo will die.

A Doggo is represented by 3 characteristics:

* Food
* Fat
* Affection

All of these charactesristics are between 0 and 1 and help to determine the happiness of the Doggo:

$$happiness = min(food, 1-fat, affection)$$

Once the happiness reaches 0, the Doggo dies.

Food and affection will diminish in time. In my test, it decreases by a value of 0.01 every minute. Fat doesn't change with time. However, action can be taken to improve the levels of food and affection, or reduce fat.

Read more [here](src/data/README.md).


## Reinforcement Learning

The primary goal of this project is to be able to make Doggos immortal. We want to find a modelthat will take the correct action to ensure that the Doggo will continue to live forever, with no restrictions on the number of actions.

(Note: later, I want to optimize the number of actions taken by the model, and optimize a long period without actions).

### State - Action - Reward

#### State

A state is define as the combination of:

* the 3 characteristics (food, fat, affection)
* a boolean, whether an action can be taken

#### Action

3 actions are possible: feeding, playing and walking. Feeding will affect positively the food, and negatively the fat ; Playing and walking will affect positively the fat and the affection.

|            |       | Effect on |           | Duration before effect | 
| :--------- | :---: | :-------: | :-------: | ---------------------: | 
| **Action** | Food  | Fat       | Affection |                        | 
| feed       | +25   | +25       | 0         | 1                      | 
| play       | 0     | -12       | +5        | 4                      | 
| feed       | 0     | -20       | +40       | 15                     | 


It is important to notice that, even though walking has a greater effect on the characteristics, it takes a longer time to have an effect. Taking your Doggo for a walk when it only has 5 minutes left to live will have no effect at all.

#### Reward

As a reward, we conside the happiness of the new state (therefore between 0 and 1). A (negative) reward of $-10$  is applied of the Doggo dies.


### Implemented technics

* [SARSA](src/models/value_based/sarsa/README.md)

The first technic added is called SARSA (= State Action Reward next State next Action). 

In the future:

* Monte Carlo Process
* Q-Learning
* Double Q-Learning

More information can be found to the specific documentation for each method, as well as how to use it.

## How to use it (simplified)

### Launch a SARSA test

Read more [here](src/models/value_based/sarsa/README.md).

~~~python
train_model_SARSA.py [-h] [--render_episode [RENDER_EPISODE]]
                            [--run_episode RUN_EPISODE]
                            [--update_episode_division {0,1,5,10,20,50}]
                            [--play_music [PLAY_MUSIC]]
                            method_id
~~~

### Current process

Read more [here](src/visualisation/README.md).

~~~
python3 src/visualization/app_currently_running.py
~~~
~~~
http://127.0.0.1:8052/
~~~


### Analysis

Read more [here](src/visualisation/README.md).

~~~
python3 src/visualization/app_analysis.py
~~~
~~~
http://127.0.0.1:8051/
~~~


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── figures        <- Generated graphics and figures
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── log            <- Logs using a default name: 'YYYYmmdd_HHMMSS_file_handler_name'
    │   │   └── pid        <- Current processes, defined by a pid
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── unit_tests     <- unit tests
    │   │
    │   ├── utils          <- utilities, functions which can safely be used in all programs
    │   │   └── io.py
    │   │   └── logger_module.py
    │   │
    │   └── visualization  <- Scripts/Apps to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
