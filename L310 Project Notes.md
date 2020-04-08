# Project L310
## Project description
Use reinforcement learning to learn a controller for a robot to follow a path. You can use OpenAI Gym and OpenAI baseline packages. As a first step, design the path following environment, where the robot con control forwards and rotational velocities in order to follow a path. Discuss the reward function, observation space and hyper-parameters of your RL
method of choice. Extensions: Implement a second environment where the robot does not directly learn velocities, but instead, learns the hyper-parameters of a feedback-linearized lowlevel path-following controller. Analyze how these hyper-parameters change as a function of observations, and compare to the original end-to-end solution.

## Questions for first session
* Dependencies problem; can I remove ROS or will I still need it? Upgrade to 18 might help too, as it weirdly tries to load a python2 pacakge while running with python3
* Should I create a custom openai gym env like this: https://github.com/openai/gym/blob/master/docs/creating-environments.md ?
    * > Most of their environments they did not implement from scratch, but rather created a wrapper around existing environments and gave it all an interface that is convenient for reinforcement learning.

    from https://stackoverflow.com/a/45203928
    Should I do this?
* And should I just generate random paths? Any tips for defining environment? Any requirements?
* > Discuss the reward function, observation space and hyper-parameters of your RL
method of choice.
    * So I can just use an openai baslines standard algorithm and explain its properties? Or do I have to implement the algo myself?
* So use a differential drive robot?
* For the extension, I should use one of these tracking points in front of the robot? 
* Can I reuse stuff from the assignments?



## Virtualenv cmd
. ~/L310env/bin/activate

. ~/L310envnogpu/bin/activate

## Notes q&a lab session
Use Ryan's gaussian thing as the regressor function OR use a simple neural net. The idea is that you never see the exact same state-action combination twice in a continuous space. But the NN returns a reward for a state-action combination (or just state depending on the algo)

for the extension, the idea is that you take a working controller to follow the path, but learn the hyperparamters for feedback linearzation (e.g. epsilon) to that the error (distance between path and robot) is minimized. E.g. it might make epsilon (the distance between the holonomic point and the robot) smaller in corners and bigger away from corners.

Create custom openai environment. Define reward function as something like: the further away from the path, the lower the reward.

he thinks implementing it myself is more interesting for the report than using the baslines package and it shouldn't be too much work.

## Question on action space
The action space is continuous, so how to do the equivalent of an argmax over the NN outputs?

In addition, our action space is 2d. If first discretizing the action space, should I thus create new actions, which are actually pairs of ` [u,w]`?
Or can I make v the input to the environment and let the environment do the feedback_linearization?

## Experiments
* think about which path points I'm giving the robot. maybe give more, less, or depending on what's in front rather than just the closest
* larger action space experiment
* Reduce the number of actions to make it simpler or increase to make larger
* measure time and cumulative error on a test set / dev set

## Next steps:

* save dqns for later final comparison
## References
https://arxiv.org/pdf/1511.04143.pdf about continous action spaces etc
