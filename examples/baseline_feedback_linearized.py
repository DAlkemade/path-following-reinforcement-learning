import gym
# noinspection PyUnresolvedReferences
import gym_path
import numpy as np

env = gym.make("PathFollower-v0")

X = 0
Y = 1
YAW = 2
SPEED = 1.

num_states = len(env.observation_space.sample())


# TODO clean up this class or delete (probably possible by using the new gym env and sending a constant action

def feedback_linearized(pose, velocity, epsilon):
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    # MISSING: Implement feedback-linearization to follow the velocity
    # vector given as argument. Epsilon corresponds to the distance of
    # linearized point in front of the robot.
    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
    w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))
    print('u: {} w: {}'.format(u, w))

    return u, w


def get_velocity(position, path_points):
    v = np.zeros_like(position)
    # print(path_points)
    if len(path_points) == 0:
        print("Reached goal 1")
        return v
    # Stop moving if the goal is reached.
    # strip zeroes from path points
    tmp = []
    for point in path_points:
        if not np.linalg.norm(point) <= 1E-10:
            tmp.append(point)
    path_points = tmp
    if np.linalg.norm(position - path_points[-1]) < .05:
        print("Reached goal 2")
        return v
    closest = float('inf')
    closest_point_index = None
    for i, point in enumerate(path_points):
        d = np.linalg.norm(position - point)
        if d < closest:
            closest = d
            closest_point_index = i
    if closest_point_index >= len(path_points) - 1:
        return v

    delta = path_points[closest_point_index + 1] - position
    v = SPEED * delta / np.linalg.norm(delta)
    # MISSING: Return the velocity needed to follow the
    # path defined by path_points. Assume holonomicity of the
    # point located at position.
    print('V: {}'.format(v))
    return v


for i_episode in range(20):
    observation = env.reset()
    for t in range(300):
        env.render()
        print(observation)
        path_points = np.reshape(observation, (int(num_states / 2), 2))
        print(path_points)
        velocity = get_velocity([0., 0.], path_points)
        u, w = feedback_linearized([0., 0., 0.], velocity, .1)
        # action = env.action_space.sample()
        action = np.array([u, w])
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
