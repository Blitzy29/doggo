# DOGGO


Reinforcement Learning to make a Doggo immortal.


# Table of contents

* [What is a Doggo?](#what-is-a-doggo?)


## What is a Doggo?

Doggo is first a project by [Romaric Thiam](https://github.com/RTH00). Remember tamagotchi? Doggo are similar. It is a digital pet, which one has to take care of by taking some actions, otherwise the Doggo will die.

A Doggo is represented by 3 characteristics:

* Food
* Fat
* Affection

All of these charactesristics are between 0 and 1 and help to determine the happiness of the Doggo:

<p align="center">
<img src="../../reports/figures/README/doggo_happiness.png" width="350">
</p>

Once the happiness reaches 0, the Doggo dies.

Food and affection will diminish in time. In my test, it decreases by a value of 0.01 every minute. Fat doesn't change with time. However, action can be taken to improve the levels of food and affection, or reduce fat.


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

<center>

|            |       | Effect on |           | Duration before effect | 
| :--------- | :---: | :-------: | :-------: | ---------------------: | 
| **Action** | Food  | Fat       | Affection |                        | 
| feed       | +25   | +25       | 0         | 1                      | 
| play       | 0     | -12       | +5        | 4                      | 
| walk       | 0     | -20       | +40       | 15                     | 

</center>

It is important to notice that, even though walking has a greater effect on the characteristics, it takes a longer time to have an effect. Taking your Doggo for a walk when it only has 5 minutes left to live will have no effect at all.

#### Reward

As a reward, we conside the happiness of the new state (therefore between 0 and 1). A (negative) reward of $-10$  is applied of the Doggo dies.

--------
