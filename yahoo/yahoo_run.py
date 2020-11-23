# import ray
# ray.init()

user_count = 100

def run_experiment(agents, feeds, user_model, user_features, exp, num_episodes, env_type, writer=None):
    import time
    from yahoo_environment import YahooFeedUnit, YahooFeed
    import numpy as np

    cur_time = float(time.time())

#     print('num units: {}, num negative: {}, randomize: {}, experiment: {}'.format(n, n - num_positive, r, exp))

    agents_episode_reward = {}
    agents_cumulative_reward = {}
    for agent in agents:
        agents_cumulative_reward[agent.agent_name] = []
        agents_episode_reward[agent.agent_name] = []

    for j in range(num_episodes):
        if j % 1 == 0:
            print('Experiment {} at episode {}, used time {}'.format(exp, j, time.time() - cur_time))
            cur_time = time.time()

        feeds = generate_feeds(10, 4)
        for i in range(len(agents)):
            agents_episode_reward[agents[i].agent_name].append(0)
            if j == 0:
                agents_cumulative_reward[agents[i].agent_name].append(0)
            else:
                agents_cumulative_reward[agents[i].agent_name].append(
                    agents_cumulative_reward[agents[i].agent_name][j - 1])

        for k, user_feature in enumerate(user_features):
            envs = [
                YahooFeed(feeds, user_model, user_feature, env_type)
                for _ in range(len(agents))
            ]

            # for i, agent in enumerate(agents):
            #     print(agent.agent_name, agent.cum_rewards)
            #     print(envs[i].interest_level)
                # print(envs[i].current_feed)

            for i, agent in enumerate(agents):
                envs[i].reset(user_feature)
                agents[i].reset(user_feature, feeds[0], k)

            for i in range(len(agents)):
                scroll = True
                while scroll:
                    action = agents[i].choose_action()
                    scroll, reward, next_batch = envs[i].step(action)
                    agents[i].update_buffer(scroll, reward, next_batch)

                agents[i].learn_from_buffer()

                agents_episode_reward[agents[i].agent_name][j] += agents[i].cum_rewards
                agents_cumulative_reward[agents[i].agent_name][j] += agents[i].cum_rewards

        for i in range(len(agents)):
            if writer:
                writer.add_scalar('episodic_reward_' + agents[i].agent_name + '_' + str(exp), agents_episode_reward[agents[i].agent_name][j], j)
                writer.add_scalar('cumulative_reward_' + agents[i].agent_name + '_' + str(exp), agents_cumulative_reward[agents[i].agent_name][j], j)

    return agents_episode_reward

# @ray.remote
def experiment_wrapper(user_features, user_model, feed_count, i, num_episodes, experiment_name, env_type, writer=None):
    from yahoo_supervised_agent import YahooSupervisedAgent
    from yahoo_supervised_agent_one_step import YahooSupervisedAgentOneStep
    from yahoo_supervised_ncf_agent import YahooSupervisedNCFAgent
    from yahoo_supervised_ncf_agent_one_step import YahooSupervisedNCFOneStepAgent
    from yahoo_dqn_ncf_agent import YahooDQNNCFAgent
    from yahoo_dqn_agent import YahooDQNAgent
    from yahoo_deep_exp_agent import YahooDeepExpAgent
    from yahoo_deep_exp_ncf_agent import YahooDeepExpNCFAgent
    import numpy as np
    import os

    deep_exp_agents = []
    feed_units = generate_feeds(episode_length, candidate_count)
    for prior in range(0, 1):
        # deep_exp_agents.append(
        #     YahooDeepExpAgent(
        #         feed_units[0], user_features[0], feed_count,
        #         'deep_exploration_{}_{}'.format(feed_count, prior),
        #         prior_variance=10**prior,
        #         bootstrap=False,
        #     )
        # )
        deep_exp_agents.append(
            YahooDeepExpNCFAgent(
                feed_units[0], user_features[0], 0,
                feed_count, 'NCFDE',
                prior_variance=10**prior,
                bootstrap=False,
            ),
        )
    agents = [
        # YahooSupervisedAgent(feed_units[0], user_features[0], feed_count, 'boltzmann_TD1_{}'.format(feed_count)),
        # YahooSupervisedAgentOneStep(feed_units[0], user_features[0], feed_count, 'boltzmann_supervised_{}'.format(feed_count)),
        # YahooSupervisedNCFAgent(feed_units[0], user_features[0], 0, feed_count, 'boltzmann_TD1_ncf_{}'.format(feed_count)),
        # YahooSupervisedNCFOneStepAgent(feed_units[0], user_features[0], 0, feed_count, 'boltzmann_supervised_ncf_{}'.format(feed_count)),
        # YahooSupervisedAgent(feed_units[0], user_features[0], feed_count, 'TD1_{}'.format(feed_count), boltzmann=False),
        # YahooSupervisedAgentOneStep(feed_units[0], user_features[0], feed_count, 'supervised_{}'.format(feed_count), boltzmann=False),
        YahooSupervisedNCFAgent(feed_units[0], user_features[0], 0, feed_count, 'NCF TD(1)', boltzmann=False),
        # YahooSupervisedNCFOneStepAgent(feed_units[0], user_features[0], 0, feed_count, 'supervised_ncf_{}'.format(feed_count), boltzmann=False),
        # YahooDQNNCFAgent(feed_units[0], user_features[0], 0, feed_count, 'boltzmann_dqn_ncf_{}'.format(feed_count)),
        # YahooDQNAgent(feed_units[0], user_features[0], feed_count, 'boltzmann_dqn_{}'.format(feed_count)),
        YahooDQNNCFAgent(feed_units[0], user_features[0], 0, feed_count, 'NCF TD(0)', boltzmann=False),
        # YahooDQNAgent(feed_units[0], user_features[0], feed_count, 'dqn_{}'.format(feed_count), boltzmann=False),
    ] + deep_exp_agents

    cumulative_reward = run_experiment(agents, feed_units, user_model, user_features, i, num_episodes, env_type, writer)

    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
    np.save('./{}/experiment_{}_{}_{}_{}'.format(experiment_name, feed_count, feed_count, i, env_type), cumulative_reward)


def generate_feeds(episode_length, candidate_count):
    from yahoo_environment import YahooFeedUnit, YahooFeed
    import pickle as pkl
    import numpy as np

    article_count = 0

    all_article_features = pkl.load(open('all_article_features.pkl', 'rb'))
    filtered_article_features = np.array(list(
        filter(lambda x: len(x) == 6, list(all_article_features.values()))))
    selected_indices = np.random.choice(
            len(filtered_article_features),
            episode_length * candidate_count,
            replace=False)
    randomly_selected_articles = filtered_article_features[selected_indices]

    current_index = 0

    feed_units = []
    for batch_index in range(episode_length):
        current_batch = []
        for index in range(candidate_count):
            current_batch.append(
                YahooFeedUnit(
                    randomly_selected_articles[batch_index * candidate_count + index],
                    selected_indices[batch_index * candidate_count + index] + 1,
                    0,
                )
            )
        current_batch.append(YahooFeedUnit(
            np.array([0. for _ in range(6)]),
            0,
            0,
        ))
        feed_units.append(current_batch)

    return feed_units

def caller(episode_length, candidate_count, num_experiment, num_episodes, experiment_name, env_type='sparse_reward'):
    import random
    import pickle as pkl
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    all_user_features = pkl.load(open('all_user_features.pkl', 'rb'))
    user_features = all_user_features[:user_count]

    user_model = pkl.load(open('learned_user_intent_model.pkl', 'rb'))
    writer = SummaryWriter('./{}/'.format(experiment_name))

    agents_cumulative_reward = []
    agents = []

    feed_count = episode_length * candidate_count

    futures = []
    for i in range(num_experiment):
        experiment_wrapper(user_features, user_model, feed_count, i, num_episodes, experiment_name, env_type, writer)
    #     futures.append(experiment_wrapper.remote(user_features, user_model, feed_count, i, num_episodes))
    # ray.get(futures)

#     with Pool(num_experiment) as p:
#         print(p.map(experiment_wrapper, [(feed_units, i, num_episodes, randomize) for i in range(num_experiment)]))
    # writer.close()

if __name__ == '__main__':
    import os

    num_experiments = 20
    inputs = []

    num_episodes = 200
    episode_length = 10
    candidate_count = 4
    experiment_name = 'experiment_80_ncf'
    filelist = [f for f in os.listdir('./{}/'.format(experiment_name)) if f.split('.')[0] == 'events']
    for f in filelist:
        os.remove(os.path.join('./{}/'.format(experiment_name), f))
    caller(episode_length, candidate_count, num_experiments, num_episodes, experiment_name)
