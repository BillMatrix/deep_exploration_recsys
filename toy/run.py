# import ray
# ray.init()
from tensorboardX import SummaryWriter
import os

filelist = [ f for f in os.listdir('./experiment/') ]
for f in filelist:
    os.remove(os.path.join('./experiment/', f))

writer = SummaryWriter('./experiment/')

def run_experiment(agents, feeds, exp, num_episodes, r, env_type):
    import time
    from environment import FeedUnit, Feed

    cur_time = float(time.time())
    n = len(feeds)
    num_positive = 0
    for feed in feeds:
        if feed.interest > 0:
            num_positive += 1

#     print('num units: {}, num negative: {}, randomize: {}, experiment: {}'.format(n, n - num_positive, r, exp))

    envs = [
        Feed(feeds, num_positive, env_type) for _ in range(len(agents))
    ]

    agents_episode_reward = {}
    for agent in agents:
        # agents_cumulative_reward[agent.agent_name] = []
        agents_episode_reward[agent.agent_name] = []

    for j in range(num_episodes):
        if j % 100 == 0:
            print('num units: {}, num negative: {}, randomize: {}, experiment: {}'.format(n, n - num_positive, r, exp))
            print('Experiment {} at episode {}, used time {}'.format(exp, j, time.time() - cur_time))
            cur_time = time.time()
            for i, agent in enumerate(agents):
                print(agent.agent_name, agent.cum_rewards)

        for i, agent in enumerate(agents):
            envs[i].reset()
            agents[i].reset()
        for i in range(len(agents)):
            scroll = True
            # count = 0
            while scroll:
                action = agents[i].choose_action()
                # print('current_feed_interest: {}'.format(feeds[count].interest))
                # count += 1
                scroll, reward = envs[i].step(action)
                # print(reward)
                agents[i].update_buffer(scroll, reward)

            agents[i].learn_from_buffer()

            agents_episode_reward[agents[i].agent_name].append(agents[i].cum_rewards)

            if writer:
                if agents[i].agent_name != 'Oracle':
                    writer.add_scalar('running_loss_' + agents[i].agent_name + '_' + str(exp), agents[i].running_loss, j)
                writer.add_scalar('cumulative_reward_interest_unknown' + agents[i].agent_name + '_' + str(exp), agents[i].cum_rewards, j)

    return agents_episode_reward

# @ray.remote
def experiment_wrapper(feed_units, i, num_episodes, randomize, env_type):
    from supervised_agent import SupervisedAgent
    from supervised_agent_one_step import SupervisedAgentOneStep
    from dqn_agent import DQNAgent
    from deep_exp_hyper_agent import DeepExpHyperAgent
    from deep_exp_agent import DeepExpAgent
    from deep_exp_ids_agent import DeepExpIDSAgent
    import numpy as np

    deep_exp_agents = []
    num_positive = 0
    for feed in feed_units:
        if feed.interest > 0:
            num_positive += 1
    for prior in range(0, 1):
        deep_exp_agents.append(
            DeepExpAgent(
                [k for k in range(len(feed_units))],
                'deep_exploration_{}_{}_{}'.format(num_positive, len(feed_units), prior),
                prior_variance=10**prior,
            )
        )
    agents = ([
        SupervisedAgent([k for k in range(len(feed_units))], 'supervised_{}_{}'.format(num_positive, len(feed_units))),
        DQNAgent([k for k in range(len(feed_units))], 'dqn_{}_{}'.format(num_positive, len(feed_units))),
    ]
    + deep_exp_agents
    )

    cumulative_reward = run_experiment(agents, feed_units, i, num_episodes, randomize, env_type)

    np.save('ids_experiment_{}_{}_{}_{}_{}'.format(
        num_positive, len(feed_units), int(randomize), i, env_type), cumulative_reward)


def generate_feeds(n, k):
    from environment import FeedUnit, Feed

    feed_units = []
    for i in range(k):
        feed_units.append(FeedUnit(-1, i))
    for i in range(n):
        feed_units.append(FeedUnit(1, i))

    return feed_units

def caller(n, k, num_experiment, num_episodes, randomize=False, env_type='sparse_reward'):
    import random

    feed_units = generate_feeds(n, k)

    if randomize:
        random.shuffle(feed_units)

    agents_cumulative_reward = []
    agents = []

    futures = []
    for i in range(num_experiment):
        experiment_wrapper(feed_units, i, num_episodes, randomize, env_type)


if __name__ == '__main__':
    num_experiments = 5
    inputs = []

    for n in range(5, 6):
        for k in range(2, 3):
            # for r in range(0, 1):
            num_episodes = 2000
            caller(n, k, num_experiments, num_episodes, env_type='sparse_reward')

    writer.close()
