import gym
import numpy as np
import time
import copy
import os
import torch
import pickle
from spectrl.main.monitor import Resource_Model
from spectrl.rl.ars import NNPolicy, NNParams, ars, ars_delta
from spectrl.util.dist import FiniteDistribution
from spectrl.util.rl import get_rollout, RandomPolicy, print_performance, print_rollout
from collections import defaultdict, deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from joblib import dump



def generate_arrays(central_values, array_count, deviation=0.1):
   
    result = []
    for _ in range(array_count):
        # Generate random values around the central values
        random_values = np.array(central_values) + np.random.uniform(-deviation, deviation, len(central_values))
        # Append the tuple (random_values_array, empty_array) to the result list
        result.append((random_values, np.array([], dtype=float)))
    return result

class AbstractEdge:
    '''
    Class defining an abstract edge.
    Vertices are integers from 0 to |U|.

    Parameters:
        target: int (target vertex)
        predicate: state, resource -> float (predicate corresponding to target)
        constraints: list of constraints that needs to be maintained (one after the other)
    '''

    def __init__(self, target, predicate, constraints):
        self.target = target
        self.predicate = predicate
        self.constraints = constraints

    def learn_policy(self, env, hyperparams, source_vertex, target_vertex, init_dist=None, algo='ars', res_model=None, max_steps=20, safety_penalty=-1,
                     neg_inf=-10, alpha=0, use_gpu=False, render=False, spec_num = None):
        '''
        Learn policy for the abstract edge.

        Parameters:
            env: gym.Env (with additional method, set_state: np.array -> NoneType)
            init_dist: Distribution (initial state distribution)
            hyperparams: HyperParams object (corresponding to the learning algorithm)
            algo: str (RL algorithm to use)
            res_model: Resource_Model (optional)
            max_steps: int (maximum steps for an episode while training)

        Returns:
            Policy object with get_action(combined_state) function.
        '''
        if spec_num in [6]:
            gtop  = np.array([5, 8.0])
            gtop1 = np.array([10, 16.0])
            gtop2 = np.array([15, 24.0])


            if(source_vertex == 0):
                env.set_goal.startX = 0
                env.set_goal.startY = 0

                env.set_goal.endX = gtop[0] 
                env.set_goal.endY = gtop[1] 

            if(source_vertex == 1):

                env.set_goal.startX = gtop[0]
                env.set_goal.startY = gtop[1] 

                env.set_goal.endX = gtop1[0] 
                env.set_goal.endY = gtop1[1] 

            if(source_vertex == 2):

                env.set_goal.startX = gtop1[0] 
                env.set_goal.startY = gtop1[1] 

                env.set_goal.endX = gtop2[0] 
                env.set_goal.endY = gtop2[1] 


        if spec_num in [10]:


            if((source_vertex == 0) & (target_vertex == 1)):

                env.set_goal.startX = 0
                env.set_goal.startY = 16

                env.set_goal.endX = 2 
                env.set_goal.endY = 24

            if((source_vertex == 0) & (target_vertex == 2)):

                env.set_goal.startX = 0
                env.set_goal.startY = 16

                env.set_goal.endX = 8
                env.set_goal.endY = 24

            if(source_vertex == 1):

                env.set_goal.startX = 2
                env.set_goal.startY = 24

                env.set_goal.endX = 0 
                env.set_goal.endY = 32
            
            if(source_vertex == 2):

                env.set_goal.startX = 8
                env.set_goal.startY = 24

                env.set_goal.endX = 0 
                env.set_goal.endY = 32

        if spec_num in [9]:
            gt1new          =    np.array([2, 8])
            gt2new          =    np.array([8, 8])
            gfinalchoice2   =    np.array([0, 16])

            if((source_vertex == 0) & (target_vertex == 1)):

                env.set_goal.startX = 0
                env.set_goal.startY = 0

                env.set_goal.endX = 2 
                env.set_goal.endY = 8 

            if((source_vertex == 0) & (target_vertex == 2)):

                env.set_goal.startX = 0
                env.set_goal.startY = 0

                env.set_goal.endX = 8
                env.set_goal.endY = 8

            if(source_vertex == 1):

                env.set_goal.startX = 2
                env.set_goal.startY = 8

                env.set_goal.endX = 0 
                env.set_goal.endY = 16 
            
            if(source_vertex == 2):

                env.set_goal.startX = 8
                env.set_goal.startY = 8

                env.set_goal.endX = 0 
                env.set_goal.endY = 16 

        if spec_num in [7]:
            gtop  = np.array([5, 8.0])
            gtop1 = np.array([10, 16.0])
            gtop2 = np.array([15, 24.0])
            gtop3 = np.array([20, 32.0])
            gtop4 = np.array([25, 40.0])

            if(source_vertex == 0):
                env.set_goal.startX = 0
                env.set_goal.startY = 0

                env.set_goal.endX = gtop[0] 
                env.set_goal.endY = gtop[1] 

            if(source_vertex == 1):

                env.set_goal.startX = gtop[0]
                env.set_goal.startY = gtop[1] 

                env.set_goal.endX = gtop1[0] 
                env.set_goal.endY = gtop1[1] 

            if(source_vertex == 2):

                env.set_goal.startX = gtop1[0] 
                env.set_goal.startY = gtop1[1] 

                env.set_goal.endX = gtop2[0] 
                env.set_goal.endY = gtop2[1] 

            if(source_vertex == 3):

                env.set_goal.startX = gtop2[0] 
                env.set_goal.startY = gtop2[1] 

                env.set_goal.endX = gtop3[0] 
                env.set_goal.endY = gtop3[1] 

            if(source_vertex == 4):

                env.set_goal.startX = gtop3[0] 
                env.set_goal.startY = gtop3[1] 

                env.set_goal.endX = gtop4[0] 
                env.set_goal.endY = gtop4[1] 

        reach_env = ReachabilityEnv(env, init_dist, self.predicate, self.constraints, max_steps=max_steps, res_model=res_model,
                                    safety_penalty=safety_penalty, neg_inf=neg_inf, alpha=alpha)
        
        
        print("\nReach info(1) :",reach_env.get_state())
        
        # Step 2: Call the learning algorithm
        print('\nLearning policy for edge {} -> {}\n'.format(source_vertex, self.target))

        if self.constraints[0].__name__ == 'true_pred' and self.predicate is None:
            policy = RandomPolicy(reach_env.action_space.shape[0], reach_env.action_space.high)
            log_info = np.array([[0, 0, 0]])
        elif algo == 'ars':
            nn_params = NNParams(reach_env, hyperparams.hidden_dim)
            policy = NNPolicy(nn_params)
            log_info = ars(reach_env, policy, hyperparams.ars_params, spec_num = spec_num)
        elif algo == 'ddpg':
            agent = DDPG(hyperparams, use_gpu=use_gpu)
            agent.train(reach_env)
            policy = agent.get_policy()
            log_info = agent.rewardgraph
        else:
            raise ValueError('Algorithm \"{}\" not supported!'.format(algo))

        # Render for debugging
        if render:
            for _ in range(20):
                get_rollout(reach_env, policy, True)
            reach_env.close_viewer()

        return policy, reach_env, log_info

def initialize_reach_probability(edges, n):
    
    unique_vertices = set()
    for edge in edges:
        unique_vertices.add(edge[0])  # Source vertex
        unique_vertices.add(edge[1])  # Target vertex

    # Initialize dictionary with each vertex as key and a list of n zeros as value
    reach_probability_vertex = {vertex: [0] * n for vertex in unique_vertices}
    return reach_probability_vertex

def initialize_best_edge(edges, n):

    unique_vertices = set()
    for edge in edges:
        unique_vertices.add(edge[0])  # Source vertex
        unique_vertices.add(edge[1])  # Target vertex

    # Initialize dictionary with each vertex as key and a list of n zeros as value
    best_incoming_edge = {vertex: [None] * n for vertex in unique_vertices}
    return best_incoming_edge

def initialize_mid(edges, n):

    unique_vertices = set()
    for edge in edges:
        unique_vertices.add(edge[0])  # Source vertex
        unique_vertices.add(edge[1])  # Target vertex

    # Initialize dictionary with each vertex as key and a list of n zeros as value
    mid = {vertex: [[]] * n for vertex in unique_vertices}
    return mid

def build_incoming_edges_dict(edges):

    incoming_edges = {}
    for source, target in edges:

        if target not in incoming_edges:
            incoming_edges[target] = []

        incoming_edges[target].append(source)

        if source not in incoming_edges:
            incoming_edges[source] = []

    return incoming_edges

def build_outgoing_edges_dict(edges):
    
    outgoing_edges = {}

    for source, target in edges:

        if source not in outgoing_edges:
            outgoing_edges[source] = []

        outgoing_edges[source].append(target)

        if target not in outgoing_edges:
            outgoing_edges[target] = []

    return outgoing_edges

def find_initial_vertices(edges):
    
    all_vertices = set()
    vertices_with_incoming_edges = set()

    for source, target in edges:
        all_vertices.add(source)
        all_vertices.add(target)
        vertices_with_incoming_edges.add(target)

    initial_vertices = all_vertices - vertices_with_incoming_edges
    return initial_vertices

def find_final_vertices(edges):

    all_vertices = set()
    vertices_with_outgoing_edges = set()

    for source, target in edges:
        all_vertices.add(source)
        all_vertices.add(target)
        # Exclude self-loops from vertices with outgoing edges
        if source != target:
            vertices_with_outgoing_edges.add(source)

    final_vertices = all_vertices - vertices_with_outgoing_edges
    return final_vertices


def initialize_decision_set(edges):

    unique_vertices = set()
    for edge in edges:
        unique_vertices.add(edge[0])  # Source vertex
        unique_vertices.add(edge[1])  # Target vertex

    # Initialize dictionary with each vertex as key and a list of n zeros as value
    decision_sets = {vertex: [] for vertex in unique_vertices}
    return decision_sets


def deltaEdgePolicy(absReach ,edge, base_policy, all_environments, hyperparams, source_vertex, target_vertex, init_dist=None, algo='ars', res_model=None, max_steps=20,safety_penalty=-1, neg_inf=-10, alpha=0, use_gpu=False, render=False, train_nrounds=0, jump=2, nkappa=1, imgdir = None, addlogdir = None, spec_num = None):
    # train_nrounds = len(all_environments)
    all_reach_env = [None] * train_nrounds
    all_environments.set_goal.edg_num = source_vertex
    all_environments.set_goal.target_num = target_vertex
    
    for i in range(0, train_nrounds, 1):
        
        all_environments.set_goal.round_num = i+1
        absReach.update()

        print("Env goals :", all_environments.set_goal.startX, all_environments.set_goal.startY, all_environments.set_goal.endX, all_environments.set_goal.endY)
 
        all_reach_env[i] = ReachabilityEnv(all_environments, init_dist[i], edge.predicate, edge.constraints,
                                    max_steps=max_steps, res_model=res_model, safety_penalty=safety_penalty, neg_inf=neg_inf, alpha=alpha)
        
        
        print("Reach info (deltaEdgePolicy) :", all_reach_env[i].reset())

    # Step 2: Call the learning algorithm
    print('\nLearning policy for edge {} -> {}\n'.format(source_vertex, edge.target))

    if edge.constraints[0].__name__ == 'true_pred' and edge.predicate is None:
        # kappa = RandomPolicy(all_reach_env[0].action_space.shape[0], all_reach_env[0].action_space.high)
        kappas = [RandomPolicy(all_reach_env[0].action_space.shape[0], all_reach_env[0].action_space.high) for _ in range(nkappa)]
        
        all_log_info = np.array([ [[0, 0, 0]]  for i in range(train_nrounds)])
    elif algo == 'ars':
        nn_param = NNParams(all_reach_env[0], hyperparams.hidden_dim)
        
        kappas = [NNPolicy(nn_param) for _ in range(nkappa)]
        
        all_log_info = ars_delta(absReach ,all_reach_env, kappas, hyperparams.ars_params, base_policy, source_vertex, edge.target, jump, imgdir = imgdir, addlogdir = addlogdir, spec_num = spec_num)
    else:
        raise ValueError('Algorithm \"{}\" not supported!'.format(algo))

    return kappas, all_reach_env, all_log_info

def TrainKappa(absReach, base_nn_policy, env, hyperparams, algo='ars', res_model=None, max_steps=20, safety_penalty=-1, neg_inf=-10, alpha=0, use_gpu=False,
               render=False, num_samples = 100, succ_thresh = 0, abstract_policy_t = [], save_path = "", train_nrounds=0, jump=2, nkappa=1, imgdir = None, addlogdir = None, spec_num = None):

    saved_rollouts = {}
    saved_actions = {}

    print(f"Env info (TrainKappa) : {env.state}")
    
    edges = []
    for i in range(absReach.num_vertices):
        for edge in absReach.abstract_graph[i]:
            edges.append([i, edge.target])

    incoming = build_incoming_edges_dict(edges)
    initial_vertices = find_initial_vertices(edges)

    queue = deque()
    queue.extend(initial_vertices)
    print('Initial Queue: ', queue)

    abstract_policy = [-1] * absReach.num_vertices
    kappa = [[] for _ in absReach.abstract_graph]
    assert len(kappa) == len(base_nn_policy) 

    
    # Reach states for each vertex and source
    reach_states = [{} for _ in range(train_nrounds)]
    num_edges_learned = 0
    total_steps = 0
    total_time = 0.

    reach_probability_edge = dict()
    reach_probability_vertex = initialize_reach_probability(edges, train_nrounds)
    best_incoming_vertex = initialize_best_edge(edges, train_nrounds)
    parent = [-1] * absReach.num_vertices

    
    for initial_vertex in initial_vertices:
        reach_probability_vertex[initial_vertex] = [1] * train_nrounds

    while queue:
        
        source = queue.popleft()

        # Explore the vertex by learning policies for each outgoing edge
        for e, edge in enumerate(absReach.abstract_graph[source]):
                
                target = edge.target
                parent[target] = source

                if source == 0:
                    start_dist = [None] * train_nrounds
                else:
                    start_dist = [None] * train_nrounds
                    for i in range(0, train_nrounds, 1):
                        start_dist[i] = FiniteDistribution(reach_states[i][(parent[source], source)])
                
                # Learn policy
                kappa_edge, all_reach_env, all_log_info = deltaEdgePolicy(absReach ,edge, base_nn_policy[source][e], env, hyperparams,  
                                                                            source, target, start_dist, algo, res_model, max_steps, 
                                                                            safety_penalty, neg_inf, alpha, use_gpu, render, train_nrounds, 
                                                                            jump, nkappa,  imgdir = imgdir, addlogdir = addlogdir, spec_num = spec_num)
                kappa[source].append(kappa_edge)

                num_edges_learned += 1

                total_steps, total_time  = [0.] * train_nrounds, [0.] * train_nrounds

                for i, log_info in enumerate(all_log_info):

                    if len(log_info)!=0:
                        total_steps[i] += log_info[-1][0]
                        total_time[i] += log_info[-1][1]

                # Compute reach probability and collect visited states
                reach_prob = [0.] * train_nrounds
                states_reached = [[] for _ in range(train_nrounds)]
                assert len(states_reached) == train_nrounds

                start = 0
                tr = train_nrounds

                for i in range(start, tr, 1):
                    
                    base_edge = base_nn_policy[source][e]
                    
                    round_num = (i+1)
                    rn = round_num
                    env.set_goal.round_num = round_num
                    absReach.update()

                    if ((spec_num in [8, 9, 10]) & (source == 2) & (rn >= 5)):
                        rn = round_num - 4
                    # print("\n Env info (before rollout): {} \n".format(all_reach_env[i].get_state()))
                
                    if isinstance(kappa_edge[0], NNPolicy):
                        
                        kappa_policy = NNPolicy.get_kappa(base_edge, rn, kappa_edge)


                    save_kappa = os.path.join(save_path, 'kappa_policies')

                    if not os.path.exists(save_kappa):
                        os.makedirs(save_kappa)

                    with open(f"{save_kappa}/train_kappa_policy_r{round_num}v{source}e{e}.pkl",'wb') as fp:
                        pickle.dump(kappa_policy, fp)

                    save_rollout = []
                    save_action = []
                    print("During kappa train test(after reset): {}".format(all_reach_env[i].reset()))

                    for _ in range(num_samples):
                    
                        sarss = get_rollout(all_reach_env[i], kappa_policy, False)

                        states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])

                        for state in states:
                            save_rollout.append(state[:2])
                        save_rollout.append(["separator"])

                        for action in np.array([act for _, act, _, _ in sarss]):
                            save_action.append(action)
                        save_action.append(["separator"]) 

                        total_steps[i] += len(sarss)
                        if all_reach_env[i].cum_reward(states) > 0:
                            reach_prob[i] += 1.0

                            if edge.target != source:
                                states_reached[i].append(all_reach_env[i].get_state())
                    

                    print("During kappa train test(After rollout): {}".format(all_reach_env[i].get_state()))

                    reach_prob[i] = (reach_prob[i] / num_samples)
                    saved_rollouts[f"r{round_num}v{source}e{e}"] = copy.deepcopy(save_rollout)
                    saved_actions[f"r{round_num}v{source}e{e}"] = copy.deepcopy(save_action)

                env.set_goal.round_num = 0
                    
                print('\nReach Probability: {}'.format(reach_prob))
                reach_probability_edge[(source, target)] = reach_prob

                incoming[target].remove(source)
                if (not incoming[target]): 
                    queue.append(target)

                for i in range(0, train_nrounds, 1):
                        reach_states[i][(source, edge.target)] = states_reached[i]

                if all([len(states_reached[j])> 0 for j in range(0, train_nrounds, 1)]):
                    for i in range(0, train_nrounds, 1):
                        reach_states[i][(source, edge.target)] = states_reached[i]
                    
                else:
            
                    for i in range(0, train_nrounds, 1):
                        reach_states[i][(source, edge.target)] = states_reached[i]

                    save_task = []

                    for i in range(0, train_nrounds, 1):
                        if(len(states_reached[i]) == 0):
                            save_task.append(i)

                    if spec_num in [6]:
                        shift = 0
                        if((source == 0) & (edge.target == 1)):
                            central_values = [5.5 + shift, 8]
                            for i in range(0, train_nrounds, 1):
                                if i in save_task:

                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                                    central_values[0] = central_values[0] + 0.5
                            
                        if((source == 1) & (edge.target == 2)):
                            central_values = [10.5 + shift, 16] 
                            for i in range(0, train_nrounds, 1):
                                if i in save_task:

                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                                    central_values[0] = central_values[0] + 0.5

                        if((source == 2) & (edge.target == 3)):
                            central_values = [15.5 +  shift, 24] 
                            for i in range(0, train_nrounds, 1):
                                if i in save_task:

                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                                    central_values[0] = central_values[0] + 0.5

        
                    if spec_num in [10]:

                        if((source == 0) & (edge.target == 1)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:
                                    central_values = [2, 8 + 16]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                

                        if((source == 0) & (edge.target == 2)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:
                                    central_values = [8, 8 + 16]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points

                        if((source == 1) & (edge.target == 3)):
                            for i in range(0, train_nrounds, jump):
                                
                                if i in save_task:

                                    central_values = [i + 2, 16 + 16]
                    
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                        
                        if((source == 2) & (edge.target == 3)):
                            for i in range(0, train_nrounds, jump):
                                
                                if i in save_task:

                                    central_values = [i + 2, 16 + 16]
                    
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points

                    if spec_num in [9]:

                        if((source == 0) & (edge.target == 1)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:
                                    central_values = [2, 8]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                

                        if((source == 0) & (edge.target == 2)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:
                                    central_values = [8, 8]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points

                        if((source == 1) & (edge.target == 3)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:

                                    central_values = [i + 1, 16]
                    
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                        
                        if((source == 2) & (edge.target == 3)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:

                                    central_values = [i + 1, 16]
                    
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points

                    if spec_num in [8]:

                        if((source == 0) & (edge.target == 1)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:
                                    central_values = [2, 8]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                

                        if((source == 0) & (edge.target == 2)):
                            for i in range(0, train_nrounds, 1):
                                
                                if i in save_task:
                                    central_values = [8, 8]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points

                        if((source == 1) & (edge.target == 3)):
                            for i in range(0, train_nrounds, jump):
                                
                                if i in save_task:
                                    central_values = [5, 16]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points
                        
                        if((source == 2) & (edge.target == 3)):
                            for i in range(0, train_nrounds, jump):
                                
                                if i in save_task:
                                    central_values = [5, 16]  # Replace with the central values you want
                                    array_count = num_samples
                                    points = generate_arrays(central_values, array_count, deviation=1)

                                    reach_states[i][(source, edge.target)] = points

                                       
                    if spec_num in [7]:
                        shift = 0
                        if((source == 0) & (edge.target == 1)):
                            central_values = [5.5 + shift, 8]
                            for i in range(0, train_nrounds, 1):
                            
                                array_count = num_samples
                                points = generate_arrays(central_values, array_count, deviation=1)

                                reach_states[i][(source, edge.target)] = points
                                central_values[0] = central_values[0] + 0.5
                            
                        if((source == 1) & (edge.target == 2)):
                            central_values = [10.5 + shift, 16] 
                            for i in range(0, train_nrounds, 1):

                                array_count = num_samples
                                points = generate_arrays(central_values, array_count, deviation=1)

                                reach_states[i][(source, edge.target)] = points
                                central_values[0] = central_values[0] + 0.5

                        if((source == 2) & (edge.target == 3)):
                            central_values = [15.5 +  shift, 24] 
                            for i in range(0, train_nrounds, 1):

                                array_count = num_samples
                                points = generate_arrays(central_values, array_count, deviation=1)

                                reach_states[i][(source, edge.target)] = points
                                central_values[0] = central_values[0] + 0.5

                        if((source == 3) & (edge.target == 4)):
                            central_values = [20.5 + shift, 32] 
                            for i in range(0, train_nrounds, 1):

                                array_count = num_samples
                                points = generate_arrays(central_values, array_count, deviation=1)

                                reach_states[i][(source, edge.target)] = points
                                central_values[0] = central_values[0] + 0.5
                        
                        if((source == 4) & (edge.target == 5)):
                            central_values = [25.5 + shift , 40]
                            for i in range(0, 20, 1):

                                array_count = num_samples
                                points = generate_arrays(central_values, array_count, deviation=1)

                                reach_states[i][(source, edge.target)] = points
                                central_values[0] = central_values[0] + 0.5
                        

    
                for i in range(0, train_nrounds, jump):
                    if(reach_probability_vertex[source][i] * reach_probability_edge[(source, target)][i] >
                    reach_probability_vertex[target][i]):
                        reach_probability_vertex[target][i] = reach_probability_vertex[source][i] * reach_probability_edge[(source, target)][i]
                        best_incoming_vertex[target][i] = {source}
                    elif(reach_probability_vertex[source][i] * reach_probability_edge[(source, target)][i] ==
                    reach_probability_vertex[target][i]):
                        if(best_incoming_vertex[target][i] == None):
                            pass
                        else:
                            best_incoming_vertex[target][i].union({source})            

    actions_rollouts = os.path.join(save_path, 'actions_rollouts')

    if not os.path.exists(actions_rollouts):
        os.makedirs(actions_rollouts)

    with open(f"{actions_rollouts}/train_rollouts.pkl","wb") as fp:
        pickle.dump(saved_rollouts, fp)
    
    with open(f"{actions_rollouts}/train_actions.pkl","wb") as fp:
        pickle.dump(saved_actions, fp)

    with open(f"{addlogdir}/reach_probability_vertex.pkl","wb") as fp:
        pickle.dump(reach_probability_vertex, fp)
    
    with open(f"{addlogdir}/reach_probability_edge.pkl","wb") as fp:
        pickle.dump(reach_probability_edge, fp)

    with open(f"{addlogdir}/best_incoming_vertex.pkl","wb") as fp:
        pickle.dump(best_incoming_vertex, fp)

    return kappa


class AbstractReachability:
    '''
    Class defining the abstract reachability problem.

    Parameters:
        abstract_graph: list of list of abstract edges (adjacency list).
        final_vertices: set of int (set of final vertices).

    Initial vertex is assumed to be 0.
    '''

    def __init__(self, abstract_graph, final_vertices, initial = None, update = None, isnround = None):
        self.abstract_graph = abstract_graph
        self.final_vertices = final_vertices
        self.num_vertices = len(self.abstract_graph)
        self.initial = initial
        self.update = update
        self.isnround = isnround

    def decision_boundary(self, abs_graph = None, best_incoming_vertex = None, reach_probability_edge = None, reach_probability_vertex = None, train_nrounds=0, jump=2, save_path=None):
        
        edges = []
        for i in range(abs_graph.num_vertices):
            for edge in abs_graph.abstract_graph[i]:
                edges.append([i, edge.target])
        edges = [edge for edge in edges if edge[0] != edge[1]]
        # print(edges)

        outgoing = build_outgoing_edges_dict(edges)
        outgoing_sample = build_outgoing_edges_dict(edges)
        incoming = build_incoming_edges_dict(edges)
        final_vertices = find_final_vertices(edges)
        mid = initialize_mid(edges, train_nrounds)
        decision_sets = initialize_decision_set(edges)
        

        for i in range(0, train_nrounds, jump):
            max_prob_vertex = max(final_vertices, key=lambda v: reach_probability_vertex[v][i])
            decision_sets[max_prob_vertex].append(i)

        queue = deque()
        queue.extend(final_vertices)
        print('Initial Queue: ', queue)

        # print('Mid:                ', mid)
        print('Best Incoming Edge: ', best_incoming_vertex)
        print('Decision Set:       ', decision_sets)

        for from_vertex, to_list in best_incoming_vertex.items():
            for to_vertex, to_vertex_indices in decision_sets.items():
                if from_vertex != to_vertex:
                    # Check each index in the 'to_list' to see if it connects to 'to_vertex'
                    for index, vertices in enumerate(to_list):
                        if vertices is not None and to_vertex in vertices:
                            # Add the index to the decision set of 'from_vertex'
                            if index not in decision_sets[from_vertex]:
                                decision_sets[from_vertex].append(index)

        print(decision_sets)

        def create_datas_from_decision_set_unique(decision_set, y_coordinate_value, correlation=2):
            """
            Creates a dictionary 'datas' based on the provided decision set, y-coordinate value, and correlation.
            Ensures that each x-coordinate is unique across the decision set.

            :param decision_set: A dictionary representing the decision sets for each vertex.
            :param y_coordinate_value: The value to be used for all 'y-coordinate' entries.
            :param correlation: The value to be added to the x-coordinate for correlation.
            :return: A dictionary with 'x-coordinate', 'y-coordinate', and 'Target'.
            """
            datas = {
                'x-coordinate': [],
                'y-coordinate': [],
                'Target': []
            }

            # Keep track of used x-coordinates to avoid duplication
            used_x_coordinates = set()

            for target_vertex, indices in decision_set.items():
                for index in indices:
                    adjusted_index = index + correlation
                    if adjusted_index not in used_x_coordinates:
                        used_x_coordinates.add(adjusted_index)
                        datas['x-coordinate'].append(adjusted_index)
                        datas['y-coordinate'].append(y_coordinate_value)
                        datas['Target'].append(target_vertex)

            return datas
        
        datas = create_datas_from_decision_set_unique(decision_sets, 0)
        
        def find_first_decision_point(datas, features, target):
            """
            Constructs a DataFrame from the provided data, trains a decision tree classifier,
            and finds the first decision point.

            :param datas: Dictionary containing the data.
            :param features: List of column names to be used as features.
            :param target: Column name of the target variable.
            :return: The first decision threshold value.
            """
            # Create DataFrame from the provided data
            df = pd.DataFrame(datas)

            # Create feature and target datasets
            X = df[features]
            y = df[target]

            # Train the Decision Tree Classifier
            clf = DecisionTreeClassifier()
            clf.fit(X, y)

            # Accessing the decision tree
            tree = clf.tree_

            # Extracting the threshold values and corresponding feature indices
            decision_points = [
                (tree.feature[node], tree.threshold[node])
                for node in range(tree.node_count)
                if tree.children_left[node] != tree.children_right[node]
            ]

            # Converting feature indices to feature names
            feature_names = X.columns
            formatted_decision_points = [
                (feature_names[feature_idx], threshold)
                if feature_idx != -2 else ('leaf', None)
                for feature_idx, threshold in decision_points
            ]

            dump(clf, f'{save_path}/decision_boundary/db_model.joblib')
            # Return the first decision threshold value
            return formatted_decision_points[0][1] if formatted_decision_points else None

        db = find_first_decision_point(datas, ['x-coordinate', 'y-coordinate'], 'Target')
        print('Boundary Point (x-axis): ', db)

        with open(f"{save_path}/decision_boundary/boundary_point.pkl","wb") as fp:
            pickle.dump(db, fp)


    def learn_dijkstra_policy(self, env, abs_graph, hyperparams, algo='ars', res_model=None, max_steps=100, safety_penalty=-1, neg_inf=-10, alpha=0,
                              num_samples=300, use_gpu=False, render=False, succ_thresh=0., save_path="", train_nrounds=0, jump=2, nkappa=1, imgdir = None, addlogdir = None, spec_num = None):
    
        # Initialize abstract policy and NN policies.
        abstract_policy = [-1] * self.num_vertices
        nn_policies = [[] for _ in self.abstract_graph]


        edges = []
        for i in range(abs_graph.num_vertices):
            for edge in abs_graph.abstract_graph[i]:
                edges.append([i, edge.target])

        incoming = build_incoming_edges_dict(edges)
        initial_vertices = find_initial_vertices(edges)

        queue = deque()
        queue.extend(initial_vertices)
        print('Initial Queue: ', queue)

        reach_states = {}
        num_edges_learned = 0
        total_steps = 0
        total_time = 0.


        edge_time = []
        current_rnd_time = []
      
        if self.initial is not None:
            self.initial(env.state)
            print("Env info(1): state = ", env.state)

        saved_rollouts_base = {} 
        saved_actions_base = {}

        parent = [-1] * self.num_vertices

        while queue:

            source = queue.popleft()

            for e,edge in enumerate(self.abstract_graph[source]):

                target = edge.target
                parent[target] = source
                
                if source == 0:
                    start_dist = None
                else:
                    start_dist = FiniteDistribution(reach_states[(parent[source], source)])

                # Learn policy
                start = time.time()

                edge_policy, reach_env, log_info = edge.learn_policy(env, hyperparams, source, target, start_dist, algo, res_model,
                                    max_steps, safety_penalty, neg_inf, alpha, use_gpu, render, spec_num = spec_num)
                
                print("\nReach info(2) :",reach_env.get_state())
                nn_policies[source].append(edge_policy)

                end = time.time()
                current_rnd_time.append((end-start)/60)    

                # update stats
                num_edges_learned += 1
                total_steps += log_info[-1][0]
                total_time += log_info[-1][1]

                # Compute reach probability and collect visited states
                states_reached = []
                reach_prob = 0

                #-----------------------------------------------
                save_rollout = [] #for saving rollouts
                save_actions = [] #for saving actions
                #-----------------------------------------------

                for _ in range(num_samples): 

                    sarss = get_rollout(reach_env, edge_policy, False) 
                    states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])

                    #---------------------------------
                    for state in states:
                        save_rollout.append(state[:2])
                    save_rollout.append(["separator"])

                    for action in np.array([action for _, action, _, _ in sarss] + sarss[-1][1]):
                        save_actions.append(action)
                    save_actions.append(["separator"])
                    #---------------------------------- 

                    total_steps += len(sarss)
                    if reach_env.cum_reward(states) > 0:
                        reach_prob += 1
                        if edge.target != source:
                            states_reached.append(reach_env.get_state())

                reach_prob = (reach_prob / num_samples)

                print('\nReach Probability: {}'.format(reach_prob))
                print('---------------------------------------------------------------')

                saved_rollouts_base[f'r{0}v{source}e{e}'] = copy.deepcopy(save_rollout)
                saved_actions_base[f'r{0}v{source}e{e}'] = copy.deepcopy(save_actions)
        
                incoming[target].remove(source)
                if (not incoming[target]): 
                    queue.append(target)

                if(len(states_reached)>0): 
                    reach_states[(source, edge.target)] = states_reached
                else:
                    central_values = [8, 4]  # Replace with the central values you want
                    array_count = num_samples
                    points = generate_arrays(central_values, array_count, deviation=0.5)
                    reach_states[(source, edge.target)] = points

            
        # --------saving the actions and rollous for base round-------------------
        actions_rollouts = os.path.join(save_path, 'actions_rollouts')

        if not os.path.exists(actions_rollouts):
            os.makedirs(actions_rollouts)

        with open(f"{actions_rollouts}/base_rollouts.pkl","wb") as fp:
            pickle.dump(saved_rollouts_base, fp)

        with open(f"{actions_rollouts}/base_actions.pkl","wb") as fp:
            pickle.dump(saved_actions_base, fp)

        print('TRAINING KAPPA')
        kappa = TrainKappa(self, nn_policies, env, hyperparams, algo, res_model, 
                           max_steps, safety_penalty, neg_inf, alpha, use_gpu, render, 
                           num_samples, succ_thresh, abstract_policy, save_path, train_nrounds, 
                           jump, nkappa=nkappa, imgdir = imgdir, addlogdir = addlogdir, spec_num = spec_num)
        
        # u = vertex
        # abstract_policy[u] = u
        # while u != 0:
        #     abstract_policy[parent[u]] = u
        #     u = parent[u]

        # # Change abstract policy to refer to edge number rather than vertex number
        # for v in range(self.num_vertices):
        #     if abstract_policy[v] != -1:
        #         for i in range(len(self.abstract_graph[v])):
        #             if self.abstract_graph[v][i].target == abstract_policy[v]:
        #                 abstract_policy[v] = i
        #                 break

        # # print("Abstract policy :",abstract_policy)

        return abstract_policy, nn_policies, [total_steps, total_time, num_edges_learned, edge_time], kappa

    def pretty_print(self):
        for i in range(self.num_vertices):
            targets = ''
            for edge in self.abstract_graph[i]:
                targets += ' ' + str(edge.target)
            print(str(i) + ' ->' + targets)


class ReachabilityEnv(gym.Env):
    '''
    Product of system and resource model.
    Terminates upon reaching a goal predicate (if specified).

    Parameters:
        env: gym.Env (with set_state() method)
        init_dist: Distribution (initial state distribution)
        final_pred: state, resource -> float (Goal of the reachability task)
        constraints: Constraints that need to be satisfied (defined reward function)
        res_model: Resource_Model (optional, can be None)
        max_steps: int
        safety_penalty: float (min penalty for violating constraints)
        neg_inf: float (negative reward for failing to satisfy constraints)
        alpha: float (alpha * original_reward will be added to reward)
    '''

    def __init__(self, env, init_dist=None, final_pred=None, constraints=[],
                 max_steps=20, res_model=None, safety_penalty=-1, neg_inf=-10,
                 alpha=0):
        self.wrapped_env = env
        self.init_dist = init_dist
        self.final_pred = final_pred
        self.constraints = constraints
        self.max_steps = max_steps
        self.safety_penalty = safety_penalty
        self.neg_inf = neg_inf
        self.alpha = alpha

        # extract dimensions from env
        self.orig_state_dim = self.wrapped_env.observation_space.shape[0]
        self.action_dim = self.wrapped_env.action_space.shape[0]

        # Dummy resource model
        if res_model is None:
            def delta(sys_state, res_state, sys_action):
                return np.array([])
            res_model = Resource_Model(self.orig_state_dim, self.action_dim, 0, np.array([]), delta)
        self.res_model = res_model

        obs_dim = self.orig_state_dim + self.res_model.res_init.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,))
        self.action_space = self.wrapped_env.action_space

        # Reset the environment
        self.reset()

    def reset(self):
        self.sys_state = self.wrapped_env.reset()
        if self.init_dist is not None:
            sim_state, self.res_state = self.init_dist.sample()
            self.sys_state = self.wrapped_env.set_sim_state(sim_state)
        else:
            self.res_state = self.res_model.res_init
        self.violated_constraints = 0
        self.prev_safety_reward = self.neg_inf
        self.t = 0
        return self.get_obs()

    def step(self, action):
        self.res_state = self.res_model.res_delta(self.sys_state, self.res_state, action)
        self.sys_state, r, _, _ = self.wrapped_env.step(action)
        self.t += 1

        reward = self.reward()
        reward = reward + self.alpha * min(r, 0)
        done = self.t > self.max_steps
        if (self.final_pred is not None) and (self.violated_constraints < len(self.constraints)):
            done = done or self.final_pred(self.sys_state, self.res_state) > 0

        return self.get_obs(), reward, done, {}

    def render(self):
        self.wrapped_env.render()
        print('System State: {} | Resource State: {}'.format(
            self.sys_state.tolist(), self.res_state.tolist()))

    def get_obs(self):
        return np.concatenate([self.sys_state, self.res_state])

    def get_state(self):
        return self.wrapped_env.get_sim_state(), self.res_state

    def reward(self):
        reach_reward = 0
        if self.final_pred is not None:
            reach_reward = self.final_pred(self.sys_state, self.res_state)

        safety_reward = self.prev_safety_reward
        set_new_vc = False
        for i in range(self.violated_constraints, len(self.constraints)):
            cur_constraint_val = self.constraints[i](self.sys_state, self.res_state)
            safety_reward = max(safety_reward, cur_constraint_val)
            if not set_new_vc:
                if cur_constraint_val <= 0:
                    self.violated_constraints += 1
                else:
                    set_new_vc = True
        safety_reward = min(safety_reward, 0)
        if safety_reward < 0:
            safety_reward = min(safety_reward, self.safety_penalty)
            self.prev_safety_reward = safety_reward

        return reach_reward + safety_reward

    def cum_reward(self, states):
        reach_reward = self.neg_inf
        safety_reward = -self.neg_inf
        violated_constraints = 0
        for s in states:
            sys_state = s[:self.orig_state_dim]
            res_state = s[self.orig_state_dim:]
            if self.final_pred is not None:
                reach_reward = max(reach_reward, self.final_pred(sys_state, res_state))

            cur_safety_reward = self.neg_inf
            for i in range(violated_constraints, len(self.constraints)):
                tmp_reward = self.constraints[i](sys_state, res_state)
                if tmp_reward <= 0:
                    violated_constraints += 1
                else:
                    cur_safety_reward = tmp_reward
                    break
            safety_reward = min(safety_reward, cur_safety_reward)
        if self.final_pred is None:
            reach_reward = -self.neg_inf
        return min(reach_reward, safety_reward)

    def close_viewer(self):
        self.wrapped_env.close()


class ConstrainedEnv(ReachabilityEnv):
    '''
    Environment for the full tasks enforcing constraints on the chosen abstract path.

    Parameters:
        env: gym.Env (with set_state() method)
        init_dist: Distribution (initial state distribution)
        abstract_reach: AbstractReachability
        abstract_policy: list of int (edge to choose in each abstract state)
        res_model: Resource_Model (optional, can be None)
        max_steps: int
    '''

    def __init__(self, env, abstract_reach, abstract_policy,
                 res_model=None, max_steps=20):
        self.abstract_graph = abstract_reach.abstract_graph
        self.final_vertices = abstract_reach.final_vertices
        self.abstract_policy = abstract_policy
        super(ConstrainedEnv, self).__init__(env, max_steps=max_steps,
                                             res_model=res_model)

    def reset(self):
        obs = super(ConstrainedEnv, self).reset()
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
        self.blocked_constraints = 0
        self.update_blocked_constraints()
        return obs

    def step(self, action):
        obs, _, done, info = super(ConstrainedEnv, self).step(action)
        if self.blocked_constraints >= len(self.edge.constraints):
            return obs, 0, True, info

        if self.edge.predicate is not None:
            if self.edge.predicate(self.sys_state, self.res_state) > 0:
                self.vertex = self.edge.target
                self.edge = self.abstract_graph[self.vertex][self.abstract_policy[self.vertex]]
                self.blocked_constraints = 0

        self.update_blocked_constraints()
        if self.blocked_constraints >= len(self.edge.constraints):
            return obs, 0, True, info

        reward = 0
        if done and self.vertex in self.final_vertices:
            reward = 1
        return obs, reward, done, info

    def update_blocked_constraints(self):
        for i in range(self.blocked_constraints, len(self.edge.constraints)):
            if self.edge.constraints[i](self.sys_state, self.res_state) > 0:
                break
            self.blocked_constraints += 1


class HierarchicalPolicy:

    def __init__(self, abstract_policy, nn_policies, abstract_graph, sys_dim):
        self.abstract_policy = abstract_policy
        self.nn_policies = nn_policies
        self.abstract_graph = abstract_graph
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
        self.sys_dim = sys_dim

    def get_action(self, state):
        sys_state = state[:self.sys_dim]
        res_state = state[self.sys_dim:]
        if self.edge.predicate is not None:
            if self.edge.predicate(sys_state, res_state) > 0:
                self.vertex = self.edge.target
                self.edge = self.abstract_graph[self.vertex][self.abstract_policy[self.vertex]]
        return self.nn_policies[self.vertex][self.abstract_policy[self.vertex]].get_action(state)

    def reset(self):
        self.vertex = 0
        self.edge = self.abstract_graph[0][self.abstract_policy[0]]
