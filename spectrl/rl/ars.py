from spectrl.util.rl import get_rollout, test_policy, discounted_reward

import time
import torch
import numpy as np
import copy
import math
import pickle
import os

import matplotlib.pyplot as plt

# Parameters for training a policy neural net.
#
# state_dim: int (n)
# action_dim: int (p)
# hidden_dim: int
# dir: str
# fname: str
class NNParams:
    def __init__(self, env, hidden_dim):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high
        self.hidden_dim = hidden_dim

# Parameters for augmented random search policy.
#
# n_iters: int (ending condition)
# n_samples: int (N)
# n_top_samples: int (b)
# delta_std (nu)
# lr: float (alpha)
class ARSParams:
    def __init__(self, n_iters, n_samples, n_top_samples, delta_std, lr, min_lr, log_interval=1):
        self.n_iters = n_iters
        self.n_samples = n_samples
        self.n_top_samples = n_top_samples
        self.delta_std = delta_std
        self.lr = lr
        self.min_lr = min_lr
        self.log_interval = log_interval

# HyperParameters:
#     hidden_layer_dim = Number of neurons in the hidden layers
#     actions confined to the range [-action_bound, action_bound]
#     n_iters: int (ending condition)
#     n_samples: int (N)
#     n_top_samples: int (b)
#     delta_std (nu)
#     lr: float (alpha)
#     min_lr: float (minimum alpha)
class HyperParams:
    def __init__(self, hidden_layer_dim, n_iters, n_samples, n_top_samples, delta_std, lr, min_lr):
        self.hidden_dim = hidden_layer_dim
        self.ars_params = ARSParams(n_iters, n_samples, n_top_samples, delta_std, lr, min_lr)


# Neural network policy.
class NNPolicy:
    # Initialize the neural network.
    #
    # params: NNParams
    def __init__(self, params):
        # Step 1: Parameters
        self.params = params
        # Step 2: Construct neural network

        # Step 2a: Construct the input layer
        self.input_layer = torch.nn.Linear(self.params.state_dim, self.params.hidden_dim)

        # Step 2b: Construct the hidden layer
        self.hidden_layer = torch.nn.Linear(self.params.hidden_dim, self.params.hidden_dim)
                           
        # Step 2c: Construct the output layer
        self.output_layer = torch.nn.Linear(self.params.hidden_dim, self.params.action_dim)

        # Step 3: Construct input normalization
        self.mu = np.zeros(self.params.state_dim)
        self.sigma_inv = np.ones(self.params.state_dim)

    # Get the action to take in the current state.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
        # Step 1: Normalize state
        state = (state - self.mu) * self.sigma_inv

        # Step 2: Convert to torch
        state = torch.tensor(state, dtype=torch.float)

        # Step 3: Apply the input layer
        hidden = torch.relu(self.input_layer(state))

        # Step 4: Apply the hidden layer
        hidden = torch.relu(self.hidden_layer(hidden))
        # Step 5: Apply the output layer
        output = torch.tanh(self.output_layer(hidden))

        # Step 6: Convert to numpy
        actions = output.detach().numpy()

        # Step 7: Scale the outputs
        actions = self.params.action_bound * actions

        return actions

    # Get the best action to take in the current state.
    #
    # state: np.array([state_dim])
    def get_best_action(self, state):
        return self.get_action(state)

    # Construct the set of parameters for the policy.
    #
    # nn_policy: NNPolicy
    # return: list of torch parameters
    def parameters(self):
        parameters = []
        parameters.extend(self.input_layer.parameters())
        parameters.extend(self.hidden_layer.parameters())
        parameters.extend(self.output_layer.parameters())
        return parameters
    
    def get_copy(self):
        # new_obj = self.__init__(self.params)
        # new_obj.input_layer = copy.deepcopy(self.input_layer)
        # new_obj.hidden_layer = copy.deepcopy(self.hidden_layer)
        # new_obj.output_layer = copy.deepcopy(self.hidden_layer)
        # new_obj.mu = self.mu
        # new_obj.sigma_inv = self.sigma_inv
        # return new_obj

        return copy.deepcopy(self)

    @classmethod
    def get_kappa(cls, base_nn, rnd_no, kappa):

        num = len(kappa)
        Ni = base_nn.get_copy()

        for _ in range(rnd_no):
            Ni_1 = kappa[num-1].get_copy()
            for j in range(num-1,0,-1):
                Ni_1.input_layer.weight.data += kappa[num-j-1].input_layer.weight.data * (Ni.input_layer.weight.data ** j)
                Ni_1.input_layer.bias.data += kappa[num-j-1].input_layer.bias.data * (Ni.input_layer.bias.data ** j)

                Ni_1.hidden_layer.weight.data += kappa[num-j-1].hidden_layer.weight.data * (Ni.hidden_layer.weight.data ** j)
                Ni_1.hidden_layer.bias.data += kappa[num-j-1].hidden_layer.bias.data * (Ni.hidden_layer.bias.data ** j)

                Ni_1.output_layer.weight.data += kappa[num-j-1].output_layer.weight.data * (Ni.output_layer.weight.data ** j)
                Ni_1.output_layer.bias.data += kappa[num-j-1].output_layer.bias.data * (Ni.output_layer.bias.data ** j)
            Ni = Ni_1

        if len(kappa)==1:
            Ni = base_nn.get_copy()

            Ni.input_layer.weight.data +=  (rnd_no * (kappa[0].input_layer.weight.data))
            Ni.input_layer.bias.data += (rnd_no * (kappa[0].input_layer.bias.data))
            
            Ni.hidden_layer.weight.data += (rnd_no * (kappa[0].hidden_layer.weight.data))
            Ni.hidden_layer.bias.data += (rnd_no * (kappa[0].hidden_layer.bias.data))

            Ni.output_layer.weight.data += (rnd_no * (kappa[0].output_layer.weight.data))
            Ni.output_layer.bias.data += (rnd_no * (kappa[0].output_layer.bias.data))

        Ni.mu = kappa[0].mu
        Ni.sigma_inv = kappa[0].sigma_inv

        return Ni

def ars_delta(absReach , reach_envs, kappa, params, base_policy, source_vertex, target_vertex , jump = 2, cum_reward = False, imgdir = None, addlogdir = None, spec_num = None):
    
    if spec_num in [8, 9, 10]:
        train_nrounds = len(reach_envs)

        if ((source_vertex == 0) & (target_vertex == 1)):
            start = 0
            train_nrounds = 4
        if ((source_vertex == 0) & (target_vertex == 2)):
            start = 4
            train_nrounds = len(reach_envs)
        if ((source_vertex == 1) & (target_vertex == 3)):
            start = 0
            train_nrounds = 4
        if ((source_vertex == 2) & (target_vertex == 3)):
            start = 4
            train_nrounds = len(reach_envs)
    else:
        train_nrounds = len(reach_envs)
        print(len(reach_envs))
        # train_nrounds = 5
        start = 0

    print("kappa length : ", len(kappa))

    kappa_orig = kappa
    best_kappa = kappa
    best_reward = -1e9
    previous_best_avg = None
    old_kappa = kappa
    old_reward = -1e9
    #---------------------------------------------------------------
    reward_list = []
    success_list = []
    #---------------------------------------------------------------

    # Logging information
    log_info = [[] for _ in range(train_nrounds)]
    num_steps = [0] * train_nrounds
    start_time = time.time()

    # Step 2: Initialize state distribution estimates
    mu_sum = np.zeros(kappa[0].params.state_dim)
    sigma_sq_sum = np.ones(kappa[0].params.state_dim) * 1e-5
    n_states = 0

    steps = []

    # learning_curve_reward = []
    # learning_curve_reward = []
    # itr = [] 

    success = []
    all_reward = []

    for itr in range(params.n_iters):
        # Step 3a: Sample deltas
        deltas = []


        for _ in range(params.n_samples):
            # i) Sample delta
            delta = _sample_delta(kappa[0])
            r_plus, r_minus = [], []
            states = []

        
            for i in range(start, train_nrounds, jump):

                # ii) Construct perturbed policies
                nn_kappa_plus = [_get_delta_policy(kappa[j], delta, params.delta_std) for j in range(len(kappa))]
                nn_kappa_minus = [_get_delta_policy(kappa[j], delta, -params.delta_std) for j in range(len(kappa))]
                

                round_num = (i+1)
                rn = round_num
                # print(i)
                reach_envs[i].wrapped_env.set_goal.round_num = round_num
                absReach.update()

                if ((spec_num in [12, 13]) & (source_vertex == 2)):
                    rn = round_num - 4
                
                if ((source_vertex == 1) & (target_vertex == 3)): 
                    nn_kappa_plus = NNPolicy.get_kappa(base_policy, rn, nn_kappa_plus)
                    nn_kappa_minus = NNPolicy.get_kappa(base_policy, rn, nn_kappa_minus)
                elif((source_vertex == 2) & (target_vertex == 3)): 
                    nn_kappa_plus = NNPolicy.get_kappa(base_policy, rn, nn_kappa_plus)
                    nn_kappa_minus = NNPolicy.get_kappa(base_policy, rn, nn_kappa_minus)
                else: 
                    nn_kappa_plus = NNPolicy.get_kappa(base_policy, rn, nn_kappa_plus)
                    nn_kappa_minus = NNPolicy.get_kappa(base_policy, rn, nn_kappa_minus)
                

                # iii) Generate rollouts
                sarss_plus = get_rollout(reach_envs[i], nn_kappa_plus, False)
                sarss_minus = get_rollout(reach_envs[i], nn_kappa_minus, False)
            
                num_steps[i] += (len(sarss_plus) + len(sarss_minus))

                # iv) Estimate cumulative rewards
                if not cum_reward:
                    r_plus.append(discounted_reward(sarss_plus, 1))
                    r_minus.append(discounted_reward(sarss_minus, 1))
                else:
                    r_plus.append(reach_envs[i].cum_reward([s for s, _, _, _ in sarss_plus] + [sarss_plus[-1][-1]]))
                    r_minus.append(reach_envs[i].cum_reward([s for s, _, _, _ in sarss_minus] + [sarss_minus[-1][-1]]))

                states.append(np.array([state for state, _, _, _ in sarss_plus + sarss_minus]))
            
                reach_envs[i].wrapped_env.set_goal.round_num = 0

            def softmin(rewards, beta):
                sum_exp = sum([math.exp(-beta * r) for r in rewards])
                
                # Return the Softmin value
                return -beta * math.log(sum_exp)

            beta = 1.0 
            r_plus[0] = softmin(r_plus, beta)
            r_minus[0] = softmin(r_minus, beta)
          
            # for i in range(1, len(r_plus)):
            #     r_plus[0] += r_plus[i]
            #     r_minus[0] += r_minus[i]
                
            # r_plus[0] = r_plus[0]/len(r_plus)
            # r_minus[0] = r_minus[0]/len(r_minus)

            # v) Save delta
            deltas.append((delta, r_plus[0], r_minus[0]))

            # v) Update estimates of normalization parameters
            temp_states = copy.deepcopy(states[0])
            for state in states[1:]:
                temp2 = copy.deepcopy(state)
                temp_states = np.concatenate((temp_states, temp2),axis=0)
            states = temp_states

            mu_sum += np.sum(states)
            sigma_sq_sum += np.sum(np.square(states)) 
            n_states += len(states)
    
        steps.append(sum(num_steps))
        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape) for delta_cur in deltas[0][0]]

        for j in range(params.n_top_samples):
            # i) Unpack values
            delta, r_plus, r_minus = deltas[j]

            # ii) Add delta to the sum
            for k in range(len(delta_sum)):
                delta_sum[k] += (r_plus - r_minus) * delta[k]

        # Step 3d: Compute standard deviation of rewards
        sigma_r = np.std([delta[1] for delta in deltas] + [delta[2] for delta in deltas])

        # Step 3e: Compute step length
        delta_step = [(params.lr * params.delta_std / (params.n_top_samples * sigma_r + 1e-8))
                      * delta_sum_cur for delta_sum_cur in delta_sum]

        # Step 3f: Update kappa weights
        # kappa = _get_delta_policy(kappa, delta_step, 1.0)

        for j in range(len(kappa)):
            kappa[j] = _get_delta_policy(kappa[j], delta_step, 1.0)

        # Step 3g: Update normalization parameters
        kappa[0].mu = mu_sum / n_states
        kappa[0].sigma_inv = 1.0 / np.sqrt(sigma_sq_sum / n_states)

        # Step 3h: Logging
        if itr % 1 == 0:
            avg_reward, succ_rate = [], []
            for i in range(start, train_nrounds, jump):
                # kappa_policy = copy.deepcopy(kappa)
                round_num = (i+1)
                rn = round_num

                reach_envs[i].wrapped_env.set_goal.round_num = round_num
                absReach.update()
                
                if ((spec_num in [12, 13]) & (source_vertex == 2)):
                    rn = round_num - 4

                kappa_policy = NNPolicy.get_kappa(base_policy, rn, kappa)

                a, _ = test_policy(reach_envs[i], kappa_policy, 100, use_cum_reward=False)
                avg_reward.append(a)
                _, b = test_policy(reach_envs[i], kappa_policy, 100, use_cum_reward=True)
                succ_rate.append(b)

                time_taken = (time.time() - start_time)/60
                # print("\nEnv goals :", reach_envs[i].wrapped_env.set_goal.startX, reach_envs[i].wrapped_env.set_goal.startY, reach_envs[i].wrapped_env.set_goal.endX, reach_envs[i].wrapped_env.set_goal.endY)
                print('\nSteps taken at iteration {} and round {}: {}'.format(itr, i, num_steps[i]))
                print('Time taken at iteration {} and round {}: {} mins'.format(itr, i, time_taken))
                print('Expected reward at iteration {} and round {}: {}'.format(itr, i, a))
               
                
                reach_envs[i].wrapped_env.set_goal.round_num = 0
            
           
            avg = avg_reward

            # MEAN REWARD
            # avg_reward = sum(avg_reward)/len(avg_reward)

            # MIN REWARD
            # avg_reward = min(avg_reward)

            # SOFTMIN
            def softmin(rewards, beta):
                sum_exp = sum([math.exp(-beta * r) for r in rewards])
                
                # Return the Softmin value
                return -beta * math.log(sum_exp)

            beta = 1.0 
            avg_reward = softmin(avg_reward, beta)


            current_avg = sum(succ_rate) / len(succ_rate)
            
            if previous_best_avg is None or current_avg >= previous_best_avg:
                previous_best_avg = current_avg
                print('New High Succ Prob: ',previous_best_avg)
                for j in range(len(kappa)):
                    best_kappa[j] = kappa[j]
                print('Best Kappa Saved')
                    
            print('Best Succ Prob: ', previous_best_avg)
            

            reward_list.append(avg_reward)
            success_list.append(current_avg)
            success.append(succ_rate)
        
        all_reward.append(avg)

        data = all_reward
        transposed_data = list(zip(*data))
        plt.figure(figsize=(8,6))
        for idx, y_values in enumerate(transposed_data, 1):
            x_values = list(range(1, len(y_values)+1))
            plt.plot(x_values, y_values, '-o', label=f'Env {idx}')
        plt.xlabel('Iterations')
        plt.ylabel('Average Reward (Softmin)')
        plt.title('Learning Curve (Reward)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{imgdir}_rewards/Learning_Curve_Reward_{source_vertex}_{target_vertex}_full.png", dpi=300)
        plt.close()

        data = success
        transposed_data = list(zip(*data))
        plt.figure(figsize=(8,6))
        for idx, y_values in enumerate(transposed_data, 1):
            x_values = list(range(1, len(y_values)+1))
            plt.plot(x_values, y_values, '-o', label=f'Env {idx}')
        plt.xlabel('Iterations')
        plt.ylabel('Success Probability')
        plt.ylim([-0.2,1.1])
        plt.title('Learning Curve (Success Probability)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{imgdir}_rewards/Learning_Curve_Success_Probability_{source_vertex}_{target_vertex}_full.png", dpi=300)
        plt.close()


        plt.figure(figsize=(8,6))   
        plt.plot(steps, reward_list, '-o', label=f'Env {idx}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Rewards vs Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{imgdir}_rewards/Reward_vs_Steps_{source_vertex}_{target_vertex}_full.png", dpi=300)
        plt.close()

        data = all_reward
        transposed_data = list(zip(*data))
        for idx, y_values in enumerate(transposed_data, 1):
            plt.figure(figsize=(8,6))        
            x_values = list(range(1, len(y_values)+1))
            plt.plot(x_values, y_values, '-o', label=f'Value {idx}')
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward (Softmin)')
            plt.title(f'Learning Curve (Reward) Reach Env: {idx}')
            plt.legend()
            plt.grid(True)
            filename = f"{imgdir}_rewards/Learning_Curve_Reward_Reach_Env_{idx}_Edge_{source_vertex}_{target_vertex}.png"
            plt.savefig(filename, dpi=300)            
            plt.close()  

        data = success
        transposed_data = list(zip(*data))
        for idx, y_values in enumerate(transposed_data, 1):
            plt.figure(figsize=(8,6))        
            x_values = list(range(1, len(y_values)+1))
            plt.plot(x_values, y_values, '-o', label=f'Value {idx}')
            plt.xlabel('Iterations')
            plt.ylabel('Success Probability')
            plt.title(f'Learning Curve (Success Probability) Reach Env: {idx}')
            plt.legend()
            plt.ylim([-0.2,1.1])
            plt.grid(True)
            filename = f"{imgdir}_rewards/Learning_Curve_Success_Probability_Reach_Env_{idx}_Edge_{source_vertex}_{target_vertex}.png"
            plt.savefig(filename, dpi=300)            
            plt.close()  

        for j in range(len(kappa)):
            kappa[j] = best_kappa[j]

        for j in range(len(kappa)):
            for kappa_param, kappa_orig_param in zip(kappa[j].parameters(), kappa_orig[j].parameters()):
                kappa_orig_param.data.copy_(kappa_param.data)
            
        kappa_orig[0].mu = kappa[0].mu
        kappa_orig[0].sigma_inv = kappa[0].sigma_inv  

        if len(reward_list) > 10:
                curr_avg = np.mean(reward_list[-10:])
                prev_avg = np.mean(reward_list[-11:-1])
                if abs(curr_avg-prev_avg) < 0.0001:
                    print("Early break -> Convergence Failed")
                    break

        if len(success_list) > 10:
            # Check the last 5 values of the list
            last_five_values = success_list[-3:]
            
            # Check if all values in last_five_values are equal and greater than 0.8
            if all(x >= 0.9 and x == last_five_values[0] for x in last_five_values):
                print("Early break -> Convergence achieved")
                break 
            # if all(x >= 0.9 for x in last_five_values):
            #     print("Early break -> Convergence achieved")
            #     break 
    

    y_values = reward_list
    x_values = list(range(len(reward_list)))

    # Plotting
    plt.figure(figsize=(8, 6))  
    plt.plot(x_values, y_values, '-o')
    plt.title(f'Learning Curve Aggregate Reward Edge:{source_vertex}->{target_vertex} (Softmin)')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')

    # Save the figure
    plt.savefig(f"{imgdir}_rewards/Learning_Curve_Aggregate_Reward_Softmin_{source_vertex}_{target_vertex}.png") 
    plt.close()

    rewards_save = os.path.join(addlogdir, 'rewards')
    success_save = os.path.join(addlogdir, 'success_rates')
    steps_save = os.path.join(addlogdir, 'steps')

    if not os.path.exists(rewards_save):
        os.makedirs(rewards_save)

    if not os.path.exists(success_save):
        os.makedirs(success_save)
    
    if not os.path.exists(steps_save):
        os.makedirs(steps_save)

    with open(f"{rewards_save}/success_rates_{source_vertex}_{target_vertex}.pkl", "wb") as file:
        pickle.dump(all_reward, file)

    with open(f"{success_save}/rewards_{source_vertex}_{target_vertex}.pkl", "wb") as file:
        pickle.dump(success, file)
    
    with open(f"{steps_save}/rewards_{source_vertex}_{target_vertex}.pkl", "wb") as file:
        pickle.dump(steps, file)


    return log_info

def ars(env, nn_policy, params, cum_reward=False, spec_num = None):
    # Step 1: Save original policy
    nn_policy_orig = nn_policy
    best_policy = nn_policy
    best_reward = -1e9
    #---------------------------------------------------------------
    reward_list = []
    #---------------------------------------------------------------

    # Logging information
    log_info = []
    num_steps = 0
    start_time = time.time()

    # Step 2: Initialize state distribution estimates
    mu_sum = np.zeros(nn_policy.params.state_dim)
    sigma_sq_sum = np.ones(nn_policy.params.state_dim) * 1e-5
    n_states = 0

    # Step 3: Training iterations
    for i in range(params.n_iters):
        # Step 3a: Sample deltas
        deltas = []
        for _ in range(params.n_samples):
            # i) Sample delta
            delta = _sample_delta(nn_policy)

            # ii) Construct perturbed policies
            nn_policy_plus  = _get_delta_policy(nn_policy, delta, params.delta_std)
            nn_policy_minus = _get_delta_policy(nn_policy, delta, -params.delta_std)

            # iii) Get rollouts
            sarss_plus  = get_rollout(env, nn_policy_plus, False)
            sarss_minus = get_rollout(env, nn_policy_minus, False)
            num_steps   += (len(sarss_plus) + len(sarss_minus))

            # iv) Estimate cumulative rewards
            if not cum_reward:
                r_plus  = discounted_reward(sarss_plus, 1)
                r_minus = discounted_reward(sarss_minus, 1)
            else:
                r_plus  = env.cum_reward([s for s, _, _, _ in sarss_plus] + [sarss_plus[-1][-1]])
                r_minus = env.cum_reward([s for s, _, _, _ in sarss_minus] + [sarss_minus[-1][-1]])

            # v) Save delta
            deltas.append((delta, r_plus, r_minus))

            # v) Update estimates of normalization parameters
            states = np.array([state for state, _, _, _ in sarss_plus + sarss_minus])
            mu_sum += np.sum(states)
            sigma_sq_sum += np.sum(np.square(states))
            n_states     += len(states)

        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape) for delta_cur in deltas[0][0]]
        for j in range(params.n_top_samples):
            # i) Unpack values
            delta, r_plus, r_minus = deltas[j]

            # ii) Add delta to the sum
            for k in range(len(delta_sum)):
                delta_sum[k] += (r_plus - r_minus) * delta[k]

        # Step 3d: Compute standard deviation of rewards
        sigma_r = np.std([delta[1] for delta in deltas] + [delta[2] for delta in deltas])

        # Step 3e: Compute step length
        delta_step = [(params.lr * params.delta_std / (params.n_top_samples * sigma_r + 1e-8))
                      * delta_sum_cur for delta_sum_cur in delta_sum]

        # Step 3f: Update policy weights
        nn_policy = _get_delta_policy(nn_policy, delta_step, 1.0)

        # Step 3g: Update normalization parameters
        nn_policy.mu = mu_sum / n_states
        nn_policy.sigma_inv = 1.0 / np.sqrt(sigma_sq_sum / n_states)

        # Step 3h: Logging
        if i % params.log_interval == 0:
            avg_reward, succ_rate = test_policy(env, nn_policy, 100, use_cum_reward=cum_reward)
            time_taken = (time.time() - start_time)/60
            print('\nSteps taken at iteration {}: {}'.format(i+1, num_steps))
            print('Time taken at iteration {}: {} mins'.format(i+1, time_taken))
            print('Expected reward at iteration {}: {}'.format(i+1, avg_reward))
            if cum_reward:
                print('Estimated success rate at iteration {}: {}'.format(i+1, succ_rate))
                log_info.append([num_steps, time_taken, avg_reward, succ_rate])
            else:
                log_info.append([num_steps, time_taken, avg_reward])

            # Step 4: Copy new weights and normalization parameters to original policy
            reward_list.append(avg_reward)

            if avg_reward >= best_reward:
                best_reward = avg_reward
                best_policy = nn_policy

            if len(reward_list)>10:
                curr_avg = np.mean(reward_list[-10:])
                prev_avg = np.mean(reward_list[-11:-1])
                # print("Reward change :",curr_avg, prev_avg)
                if abs(curr_avg-prev_avg) < 0.01:
                    print("Early break")
                    break

    nn_policy = best_policy

    for param, param_orig in zip(nn_policy.parameters(), nn_policy_orig.parameters()):
        param_orig.data.copy_(param.data)
    nn_policy_orig.mu = nn_policy.mu
    nn_policy_orig.sigma_inv = nn_policy.sigma_inv

    return log_info


# Construct random perturbations to neural network parameters.
#
# nn_policy: NNPolicy
# return: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
def _sample_delta(nn_policy):
    delta = []
    for param in nn_policy.parameters():
        delta.append(torch.normal(torch.zeros(param.shape, dtype=torch.float)))
    return delta

# Construct the policy perturbed by the given delta
#
# nn_policy: NNPolicy
# delta: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
# sign: float (should be 1.0 or -1.0, for convenience)
# return: NNPolicy
def _get_delta_policy(nn_policy, delta, sign):
    # Step 1: Construct the perturbed policy
    nn_policy_delta = NNPolicy(nn_policy.params)

    # Step 2: Set normalization of the perturbed policy
    nn_policy_delta.mu = nn_policy.mu
    nn_policy_delta.sigma_inv = nn_policy.sigma_inv

    # Step 3: Set the weights of the perturbed policy
    for param, param_delta, delta_cur in zip(nn_policy.parameters(), nn_policy_delta.parameters(),
                                             delta):
        param_delta.data.copy_(param.data + sign * delta_cur)

    return nn_policy_delta
