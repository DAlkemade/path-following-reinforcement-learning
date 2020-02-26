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