Use Ryan's gaussian thing as the regressor function OR use a simple neural net. The idea is that you never see the exact same state-action combination twice in a continuous space. But the NN returns a reward for a state-action combination (or just state depending on the algo)

for the extension, the idea is that you take a working controller to follow the path, but learn the hyperparamters for feedback linearzation (e.g. epsilon) to that the error (distance between path and robot) is minimized. E.g. it might make epsilon (the distance between the holonomic point and the robot) smaller in corners and bigger away from corners.

Create custom openai environment. Define reward function as something like: the further away from the path, the lower the reward.

he thinks implementing it myself is more interesting for the report than using the baslines package and it shouldn't be too much work.