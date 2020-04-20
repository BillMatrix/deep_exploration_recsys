# import ray
# ray.init()

def run_experiment(agents, feeds, user_model, user_features, exp, num_episodes, writer=None):
    import time
    from yahoo_environment import YahooFeedUnit, YahooFeed

    cur_time = float(time.time())
    target_interest_level = 0.8

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

        for i in range(len(agents)):
            agents_episode_reward[agents[i].agent_name].append(0)
            if j == 0:
                agents_cumulative_reward[agents[i].agent_name].append(0)
            else:
                agents_cumulative_reward[agents[i].agent_name].append(
                    sum(agents_episode_reward[agents[i].agent_name])
                    + agents_cumulative_reward[agents[i].agent_name][j - 1])

        for user_feature in user_features:
            envs = [
                YahooFeed(feeds, target_interest_level, user_model, user_feature) for _ in range(len(agents))
            ]

            # for i, agent in enumerate(agents):
            #     print(agent.agent_name, agent.cum_rewards)
            #     print(envs[i].interest_level)
                # print(envs[i].current_feed)

            for i, agent in enumerate(agents):
                envs[i].reset(user_feature)
                agents[i].reset(user_feature)

            for i in range(len(agents)):
                scroll = True
                while scroll:
                    action = agents[i].choose_action()
                    scroll, reward, next_batch = envs[i].step(action)
                    agents[i].update_buffer(scroll, reward, next_batch)

                agents[i].learn_from_buffer()

                agents_episode_reward[agents[i].agent_name][j] += reward
                agents_cumulative_reward[agents[i].agent_name][j] += reward

        for i in range(len(agents)):
            if writer:
                writer.add_scalar('running_loss_' + agents[i].agent_name + '_' + str(exp), agents[i].running_loss, j)
                writer.add_scalar('cumulative_reward_' + agents[i].agent_name + '_' + str(exp), agents_episode_reward[agents[i].agent_name][j], j)

    return agents_episode_reward

# @ray.remote
def experiment_wrapper(user_features, user_model, feed_count, i, num_episodes, experiment_name, writer=None):
    from yahoo_supervised_agent import YahooSupervisedAgent
    from yahoo_dqn_agent import YahooDQNAgent
    from yahoo_deep_exp_agent import YahooDeepExpAgent
    import numpy as np
    import os

    deep_exp_agents = []
    feed_units = generate_feeds(episode_length, candidate_count)
    for prior in range(0, 3):
        deep_exp_agents.append(
            YahooDeepExpAgent(
                feed_units[0], user_features[0], feed_count,
                'deep_exploration_{}_{}'.format(feed_count, prior),
                prior_variance=10**prior,
#                     bootstrap=False
            )
        )
    agents = [
        # OracleAgent(feed_units, session_size),
        YahooSupervisedAgent(feed_units[0], user_features[0], feed_count, 'supervised_{}'.format(feed_count)),
        YahooDQNAgent(feed_units[0], user_features[0], feed_count, 'dqn_{}'.format(feed_count))
    ] + deep_exp_agents

    cumulative_reward = run_experiment(agents, feed_units, user_model, user_features, i, num_episodes, writer)

    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
    np.save('./{}/supervised_experiment_{}_{}_{}'.format(experiment_name, feed_count, feed_count, i), cumulative_reward)


def generate_feeds(episode_length, candidate_count):
    from yahoo_environment import YahooFeedUnit, YahooFeed
    import pickle as pkl
    import numpy as np

    article_count = 0

    all_article_features = pkl.load(open('all_article_features.pkl', 'rb'))
    filtered_article_features = np.array(list(
        filter(lambda x: len(x) == 6, list(all_article_features.values()))))
    randomly_selected_articles = filtered_article_features[
        np.random.choice(
            len(filtered_article_features),
            episode_length * candidate_count)]

    current_index = 0

    feed_units = []
    for batch_index in range(episode_length):
        current_batch = []
        for index in range(candidate_count):
            current_batch.append(
                YahooFeedUnit(
                    randomly_selected_articles[batch_index * candidate_count + index],
                    0,
                )
            )
        current_batch.append(YahooFeedUnit(
            np.array([-1. for _ in range(6)]),
            0,
        ))
        feed_units.append(current_batch)

    return feed_units

def caller(episode_length, candidate_count, num_experiment, num_episodes, experiment_name):
    import random
    import pickle as pkl
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    all_user_features = pkl.load(open('all_user_features.pkl', 'rb'))
    user_features = all_user_features[:20]

    user_model = pkl.load(open('learned_user_intent_model.pkl', 'rb'))
    writer = SummaryWriter('./experiment/')

    agents_cumulative_reward = []
    agents = []

    feed_count = episode_length * candidate_count

    futures = []
    for i in range(num_experiment):
        experiment_wrapper(user_features, user_model, feed_count, i, num_episodes, experiment_name, writer)
    #     futures.append(experiment_wrapper.remote(user_features, user_model, feed_count, i, num_episodes))
    # ray.get(futures)

#     with Pool(num_experiment) as p:
#         print(p.map(experiment_wrapper, [(feed_units, i, num_episodes, randomize) for i in range(num_experiment)]))
    # writer.close()

if __name__ == '__main__':
    num_experiments = 10
    inputs = []

    num_episodes = 50
    episode_length = 10
    candidate_count = 5
    experiment_name = 'deep_exp_prior_1_10_epi_40'
    caller(episode_length, candidate_count, num_experiments, num_episodes, experiment_name)
