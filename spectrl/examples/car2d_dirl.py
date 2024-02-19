from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv, ReachabilityEnv
from spectrl.main.monitor import Resource_Model
from spectrl.main.spec_compiler import ev, alw, seq, choose, for_loop, Cons
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout, print_rollout
from spectrl.rl.ars import HyperParams
from spectrl.envs.car2d import VC_Env, envGoal
from spectrl.rl.ars import NNPolicy
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
import time
import pickle
from scipy.stats import truncnorm

num_iters = [50, 100, 200, 300]

STARTX, STARTY, ENDX, ENDY = 2, 3, 4, 5


def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])

def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate

def reach_final(env_goal ,err):
    def predicate(sys_state, res_state):
        goal = [env_goal.endX, env_goal.endY]
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate

def reach_start(env_goal ,err):
    def predicate(sys_state, res_state):
        goal = [env_goal.startX, env_goal.startY]
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate


#    obstacle: np.array(4): [x_min, y_min, x_max, y_max]

def avoid(obstacle):
    def predicate(sys_state, res_state):
        return 10 * max([obstacle[0] - sys_state[0],
                         obstacle[1] - sys_state[1],
                         sys_state[0] - obstacle[2],
                         sys_state[1] - obstacle[3]])
    return predicate

def have_fuel(sys_state, res_state):
    return res_state[0]

#---------------------------------------------------------------
#sets the car at origin
def initial(env_goal):
    def predicate():
        env_goal.startX = 0.
        env_goal.startY = 0.
    return predicate

def update_start(env_goal):
    def predicate():
        env_goal.startX = env_goal.round_num * 0.5
        env_goal.startY = env_goal.round_num * 0.
    return predicate

def update_start_choice(env_goal):
    def predicate():
        env_goal.startX = 1 + env_goal.round_num * 1
    return predicate

def update_start_choice2(env_goal):
    def predicate():

        if((env_goal.edg_num == 0) & (env_goal.target_num == 1)):

            env_goal.startX =  1 +  env_goal.round_num * 1
            env_goal.startY = 0

            env_goal.endX = 2
            env_goal.endY = 8

        if((env_goal.edg_num == 0) & (env_goal.target_num == 2)):

            env_goal.startX = 1 + env_goal.round_num * 1
            env_goal.startY = 0

            env_goal.endX = 8
            env_goal.endY = 8

        if(env_goal.edg_num in [1]):

            env_goal.startX = 2
            env_goal.startY = 8

            env_goal.endX = 1 + env_goal.round_num * 1
            env_goal.endY = 16

        if(env_goal.edg_num in [2]):

            env_goal.startX = 8
            env_goal.startY = 8

            env_goal.endX =  1 + env_goal.round_num * 1
            env_goal.endY = 16

    return predicate

def update_start_choice22(env_goal):
    def predicate():

        if((env_goal.edg_num == 0) & (env_goal.target_num == 1)):

            env_goal.startX = 1 + env_goal.round_num * 1
            env_goal.startY = 16

            env_goal.endX = 2
            env_goal.endY = 24

        if((env_goal.edg_num == 0) & (env_goal.target_num == 2)):

            env_goal.startX = 1 +env_goal.round_num * 1
            env_goal.startY = 16

            env_goal.endX = 8
            env_goal.endY = 24

        if(env_goal.edg_num in [1]):

            env_goal.startX = 2
            env_goal.startY = 24

            env_goal.endX = 1 + env_goal.round_num * 1
            env_goal.endY = 32

        if(env_goal.edg_num in [2]):

            env_goal.startX = 8
            env_goal.startY = 24

            env_goal.endX = 1 + env_goal.round_num * 1
            env_goal.endY = 32

    return predicate

def update_gtop(env_goal):
    def predicate():
        env_goal.endX = gtop_update[0] + env_goal.round_num * 0.5
        env_goal.endY = gtop_update[1] + env_goal.round_num * 0.
    return predicate

def update_gtop_start(env_goal):
    def predicate():
        env_goal.startX =  env_goal.round_num * 0.5
        env_goal.startY =  env_goal.round_num * 0.

        env_goal.endX = gtop[0] + env_goal.round_num * 0.5
        env_goal.endY = gtop[1] + env_goal.round_num * 0.
    return predicate

def update_everything(env_goal):
    def predicate():
        shift = 0

        if(env_goal.edg_num == 0):
            env_goal.startX = env_goal.round_num * 0.5 + shift
            env_goal.startY = env_goal.round_num * 0.

            env_goal.endX = gtop[0] + env_goal.round_num * 0.5 + shift
            env_goal.endY = gtop[1] + env_goal.round_num * 0.

        if(env_goal.edg_num == 1):

            env_goal.startX = gtop[0] +  env_goal.round_num * 0.5 + shift
            env_goal.startY = gtop[1] + env_goal.round_num * 0.

            env_goal.endX = gtop1[0] + env_goal.round_num * 0.5 + shift
            env_goal.endY = gtop1[1] + env_goal.round_num * 0.

        if(env_goal.edg_num == 2):

            env_goal.startX = gtop1[0] + env_goal.round_num * 0.5 + shift
            env_goal.startY = gtop1[1] + env_goal.round_num * 0.

            env_goal.endX = gtop2[0] + env_goal.round_num * 0.5 + shift
            env_goal.endY = gtop2[1] + env_goal.round_num * 0.

        if(env_goal.edg_num == 3):

            env_goal.startX = gtop2[0] + env_goal.round_num * 0.5 + shift
            env_goal.startY = gtop2[1] + env_goal.round_num * 0.

            env_goal.endX = gtop3[0] + env_goal.round_num * 0.5 + shift
            env_goal.endY = gtop3[1] + env_goal.round_num * 0.

        if(env_goal.edg_num == 4):

            env_goal.startX = gtop3[0] + env_goal.round_num * 0.5 + shift
            env_goal.startY = gtop3[1] + env_goal.round_num * 0.

            env_goal.endX = gtop4[0] + env_goal.round_num * 0.5 + shift
            env_goal.endY = gtop4[1] + env_goal.round_num * 0.
            
    return predicate

def isnround():
    def predicate(sys_state):
        return 0. if sys_state[0] < 5 else 1.  
    return predicate
#---------------------------------------------------------------

# Goals and obstacles

env_goal = envGoal()

level = 8

obs  = np.array([4.0 , 3.0, 5.0, 4.0])

obs_choice2  = np.array([4.0, 3.0 + level, 6.0 , 5.0 + level])
obs_choice3  = np.array([4.7, 0, 5.3, 8])
obs_choice4  = np.array([4.6, 2 + 16, 5.4, 6 + 16])
obs_param_1  = np.array([-1.0, -1.0, 1.0, -1.0])

gt1           =    np.array([2, 4])
gt2           =    np.array([8, 4])
gfinal        =    np.array([5, 8])

gt1l          =    np.array([2, 4 + level])
gt2l          =    np.array([8, 4 + level])
gfinall       =    np.array([5, 8 + level])

gt1new          =    np.array([2, 8])
gt2new          =    np.array([8, 8])
gfinalnew       =    np.array([5, 16])
gfinalchoice2   =    np.array([0, 16])

gtop          =    np.array([5, 8])
gtop_update   =    np.array([0, 8])

gtop  = np.array([5, 8.0])
gtop1 = np.array([10, 16.0])
gtop2 = np.array([15, 24.0])
gtop3 = np.array([20, 32.0])
gtop4 = np.array([25, 40.0])

error = 1
error2 = 0.5


# SPEC 0 - Single Edge without obstacle
spec0_ = (ev(reach(gtop, error)))
spec1_ = (ev(reach_final(env_goal, error)))
spec2_ = (ev(reach_final(env_goal, error)))

spec0 = for_loop(initial, spec0_, update_start(env_goal), isnround())
spec1 = for_loop(initial, spec1_, update_gtop(env_goal), isnround())
spec2 = for_loop(initial, spec2_, update_gtop_start(env_goal), isnround())


# SPEC 1 - Single Edge with obstacle
spec3_ = alw(avoid(obs), ev(reach(gtop, error)))
spec4_ = alw(avoid(obs), ev(reach_final(env_goal, error)))
spec5_ = alw(avoid(obs), ev(reach_final(env_goal, error)))

spec3 = for_loop(initial, spec3_, update_start(env_goal), isnround())
spec4 = for_loop(initial, spec4_, update_gtop(env_goal), isnround())
spec5 = for_loop(initial, spec5_, update_gtop_start(env_goal), isnround())



# SPEC 6 - 3-Reachability Task
spec6_ = seq(seq(alw(avoid(obs_param_1),  ev(reach_final(env_goal, error))), 
              alw(avoid(obs_param_1), ev(reach_final(env_goal, error)))), 
              alw(avoid(obs_param_1), ev(reach_final(env_goal, error))))

spec6 = for_loop(initial, spec6_, update_everything(env_goal), isnround)



# SPEC 7 - 5-Reachability Task
spec7_ = seq(seq(seq(seq(alw(avoid(obs_param_1),  ev(reach_final(env_goal, error))), 
              alw(avoid(obs_param_1), ev(reach_final(env_goal, error)))), 
              alw(avoid(obs_param_1), ev(reach_final(env_goal, error)))),
              alw(avoid(obs_param_1), ev(reach_final(env_goal, error)))),
              alw(avoid(obs_param_1), ev(reach_final(env_goal, error))))

spec7 = for_loop(initial, spec7_, update_everything(env_goal), isnround) 


# SPEC 8 - Double Choice with wall obstacle (some tasks become impossible to certain goal points).

spec8_ = seq(choose(alw(avoid(obs_choice3), ev(reach(gt1new, error))), 
                     alw(avoid(obs_choice3), ev(reach(gt2new, error)))),

            alw(avoid(obs_choice3), ev(reach(gfinalnew, error))))

spec8 = for_loop(initial, spec8_, update_start_choice(env_goal), isnround)

# SPEC 9 - Double Choice with wall obstacle (some tasks become impossible to certain goal points). - Moving Goal

spec9_ = seq(choose(alw(avoid(obs_choice3), ev(reach_final(env_goal, error))), 
                     alw(avoid(obs_choice3), ev(reach_final(env_goal, error)))),

            alw(avoid(obs_choice3), ev(reach_final(env_goal, error))))

spec9 = for_loop(initial, spec9_, update_start_choice2(env_goal), isnround)

# SPEC 10 - Double Choice with wall obstacle (some tasks become impossible to certain goal points). - Moving Goal - Level 2

spec10_ = seq(choose(alw(avoid(obs_choice4), ev(reach_final(env_goal, error))), 
                     alw(avoid(obs_choice4), ev(reach_final(env_goal, error)))),

            alw(avoid(obs_choice4), ev(reach_final(env_goal, error))))

spec10 = for_loop(initial, spec10_, update_start_choice22(env_goal), isnround)


specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9]
lb =    [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    folder = flags['folder']
    print("folder",folder)
    itno = flags['itno'] #-n
    spec_num = flags['spec_num'] #-s
    gpu_flag = flags['gpu_flag']
    log_info = []
    train_nrounds = flags['num_iter'] #-h
    sub_spec = flags['sub_spec'] #-b
    jump = int(sub_spec[-2])
    nkappa = int(folder[7])

    print(f"Training with gap {jump} and {nkappa} kappas")
    
    save_path = f"{folder}/spec{spec_num}({sub_spec}r{train_nrounds})"
    n_samples = int(flags['num_samples'])
    num_iters = [int(flags['total_iters'])]

    print(f"Training for rounds : {train_nrounds} and Sub spec : {sub_spec} Num samples : {n_samples}")

    if spec_num in [1, 4]: 
            env_goal.endX = gtop_update[0]
            env_goal.endY = gtop_update[1]
    
    if spec_num in [2, 5, 6, 7]: 
            env_goal.endX = gtop[0]
            env_goal.endY = gtop[1]

    if spec_num in [10]: 
            
            env_goal.startX = 0
            env_goal.startY = 16

    for i in num_iters:
        hyperparams = HyperParams(30, i, 20, 8, 0.05, 1, 0.2)

        print('\n**** Learning Policy for Spec {} for {} Iterations ****'.format(spec_num, i))

        # Step 1: initialize system environment
        system = VC_Env(500 ,env_goal ,std=0.05)


        system.reset()
        print("System info (1) : ",system.state)

        # Step 2 (optional): construct resource model
        resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)

        # Step 3: construct abstract reachability graph
        _, abstract_reach = automaton_graph_from_spec(specs[spec_num])

        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

        #------Just to store the abstract graph---------------------------------------------------------
        Agraph = []
        for i,elist in enumerate(abstract_reach.abstract_graph):
            temp = []
            for e in elist:
                temp.append(e.target)
            Agraph.append(temp)
        
        start = time.time()      
        #--------------------------------------------------------------------------------------
        
        testdir = os.path.join('testing_results', folder, f'spec{spec_num}({sub_spec}r{train_nrounds})')
        testdir_ar = os.path.join('testing_results', folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'action_rollouts')
        test_imgdir_rollouts = os.path.join('testing_results', folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'test_imgs_rollouts')

        logdir = os.path.join(folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'hierarchy')
        addlogdir = os.path.join(folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'logs')

        imgdir = os.path.join(folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'train_imgs')
        imgdir_rewards = os.path.join(folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'train_imgs_rewards')
        imgdir_rollouts = os.path.join(folder, f'spec{spec_num}({sub_spec}r{train_nrounds})', 'train_imgs_rollouts')
        
        actions_rollouts = os.path.join(save_path, 'actions_rollouts')

        if not os.path.exists(actions_rollouts):
            os.makedirs(actions_rollouts)

        if not os.path.exists(testdir):
            os.makedirs(testdir)

        if not os.path.exists(imgdir_rewards):
            os.makedirs(imgdir_rewards)

        if not os.path.exists(imgdir_rollouts):
            os.makedirs(imgdir_rollouts)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        if not os.path.exists(addlogdir):
            os.makedirs(addlogdir)

        if not os.path.exists(testdir_ar):
            os.makedirs(testdir_ar)

        if not os.path.exists(test_imgdir_rollouts):
            os.makedirs(test_imgdir_rollouts)

        if spec_num in [8, 9, 10]:
            desbo = os.path.join(save_path, 'decision_boundary')

            if not os.path.exists(desbo):
                os.makedirs(desbo)
                

        #-----------------------------------------------------------------------------------------------
            
        training = True
        decision_boundary = True
        rollouts_train = True
        testing = False
        rollouts_test = False

        if training:
  
            abstract_policy, nn_policies, stats, kappa = abstract_reach.learn_dijkstra_policy(system, abstract_reach,hyperparams,algo='ars', res_model=None, max_steps=20,
                neg_inf=-lb[spec_num], safety_penalty=-1, num_samples=n_samples, use_gpu=gpu_flag, render=render, save_path=save_path, train_nrounds=train_nrounds, jump=jump, nkappa=nkappa, imgdir = imgdir, addlogdir = addlogdir, spec_num = spec_num)
            
            end = time.time()
            edge_time = stats[-1]
            stats = stats[:-1]

            print("Total training time :",(end-start)/60)
            # #----------------------------------------------------------------------------------------------

            with open(f"{logdir}/NNpolicy_spec{spec_num}","wb") as fp:
                                pickle.dump(nn_policies, fp)

            with open(f"{logdir}/Apolicy_spec{spec_num}","wb") as fp:
                                pickle.dump(abstract_policy, fp)

            with open(f"{logdir}/Kappa_spec{spec_num}","wb") as fp:
                                pickle.dump(kappa, fp)

            with open(f"{logdir}/Stats{spec_num}","wb") as fp:
                                pickle.dump(stats, fp)

            print("MODELS SAVED")
        
        if decision_boundary:
            
            with open(f"{addlogdir}/best_incoming_vertex.pkl",'rb') as fp:
                best_incoming_vertex = pickle.load(fp)

            with open(f"{addlogdir}/reach_probability_edge.pkl",'rb') as fp:
                reach_probability_edge = pickle.load(fp)

            with open(f"{addlogdir}/reach_probability_vertex.pkl",'rb') as fp:
                reach_probability_vertex = pickle.load(fp)

            decision_boundary = abstract_reach.decision_boundary(abstract_reach, best_incoming_vertex, reach_probability_edge, reach_probability_vertex, train_nrounds, jump,  save_path)

        # # #--------------Load saved Policy----------------------------------------------------------------
            
        with open(f"{logdir}/NNpolicy_spec{spec_num}",'rb') as fp:
            nn_policies = pickle.load(fp)

        
        with open(f"{logdir}/Apolicy_spec{spec_num}",'rb') as fp:
            abstract_policy = pickle.load(fp)
    

        with open(f"{logdir}/Kappa_spec{spec_num}",'rb') as fp:
            kappa = pickle.load(fp)

        rnd_no = 0

        #----------Load saved action and states------------------------------------------------------------------------

        if rnd_no == 0:
            with open(f"{actions_rollouts}/base_actions.pkl",'rb') as fp:
                train_actions = pickle.load(fp)

            with open(f"{actions_rollouts}/base_rollouts.pkl",'rb') as fp:
                train_rollouts = pickle.load(fp)
        
        # ----------------------Mean of rollouts training-----------------------------------------
        if spec_num in [8, 9, 10]:
            with open(f"{save_path}/decision_boundary/boundary_point.pkl",'rb') as fp:
                db = pickle.load(fp)
        
        # db = int(db)
                
        if rollouts_train:
        
            print('PREPARING TRAIN ROLLOUTS IMAGES...')
            figure_size = (10, 8)
            if spec_num in [0, 1, 2, 3, 4, 5]:
                x_limits = (0, 10)
                y_limits = (0, 10)
            if spec_num in [9, 10]:
                x_limits = (-5, 35)
                y_limits = (-5, 50) 
            if spec_num in [7]:
                x_limits = (-5, 35)
                y_limits = (-5, 50) 
            if spec_num in [6]:
                x_limits = (-5, 20)
                y_limits = (-5, 30) 
            if spec_num in [8, 9]:
                x_limits = (0, 12)
                y_limits = (0, 20) 
            if spec_num in [10]:
                x_limits = (0, 12)
                y_limits = (0, 34) 

            # plt.figure(figsize=figure_size)
            fig1, ax1 = plt.subplots(figsize = figure_size)

            for rnd_no in range(1, train_nrounds + 1, 1):
                env_goal.round_num = rnd_no
                abstract_reach.update()

                # Load the appropriate file depending on rnd_no
                file_to_load = "base_rollouts.pkl" if rnd_no == 0 else "train_rollouts.pkl"
                with open(f"{actions_rollouts}/{file_to_load}", 'rb') as fp:
                    rollouts = pickle.load(fp)
                # Define the keys to process based on the round number
                keys_to_process = []
                
                if spec_num in [8, 9, 10]:
                    for v in range(abstract_reach.num_vertices):

                        if v in [0, 1]:
                            if rnd_no < db:          
                                keys_to_process.append(f"r{rnd_no}v{v}e0")

                        if v in [0, 2]:
                            if rnd_no >= db:          
                                keys_to_process.append(f"r{rnd_no}v{v}e1")
                                keys_to_process.append(f"r{rnd_no}v2e0")
                    
                else:
                    for v in range(abstract_reach.num_vertices):           
                        keys_to_process.append(f"r{rnd_no}v{v}e0")

                # Start a new figure for this round number
                fig, ax = plt.subplots(figsize=figure_size)

                # Process each key and plot in the same figure
                for key in keys_to_process:
                    # Check if the key exists in rollouts
                    if key not in rollouts:
                        continue  # Skip if the key does not exist

                    points = []
                    temp = []
                    nrl = 0
                    for x in rollouts[key]:
                        if 'separator' in x:
                            if not points:
                                points = temp[:]
                            else:
                                points = [x + y for x, y in zip(points, temp)]
                            temp = []
                            nrl += 1
                        else:
                            temp.append(x)

                    for i, x in enumerate(points):
                        points[i] = x / nrl

                    # Only process the points if they meet the conditions
                    processed_points = [x for x in points if -10 < x[0] < 50 and -10 < x[1] < 50]
                    if processed_points:
                        # Plot the processed points for this key
                        x, y = np.array(processed_points).T
                        ax.scatter(x, y, label=f'Key {key}')
                        ax.plot(x, y)
                        ax1.scatter(x, y, label=f'Key {key}')
                        ax1.plot(x, y)
                    

                # Finalize and save the figure with all the plots for this round number
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                ax.legend()
                ax.set_xlabel("x co-ordinate")
                ax.set_ylabel("y co-ordinate")
                ax.set_title(f'Trajectory of car in environment (Round {rnd_no})')
                # ax.grid(True)
                plt.savefig(f"{imgdir_rollouts}/round_{rnd_no}.png")
                plt.close(fig)

            
            if spec_num in [8, 9]:

                gt1new          =    np.array([2, 8])
                gt2new          =    np.array([8, 8])
                gfinalnew       =    np.array([5, 16])
                gfinalchoice2   =    np.array([0, 16])
                
                rect1 = patches.Rectangle((5, 1), 1, 7, linewidth=1, edgecolor='b', facecolor='b')
                rect2 = patches.Rectangle((0, 7.8), 1.5, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                rect3 = patches.Rectangle((9.5, 7.8), 3, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                rect4 = patches.Rectangle((4, 7.8), 3, 0.4, linewidth=1, edgecolor='b', facecolor='b')

                ax1.add_patch(rect1)
                ax1.add_patch(rect2)
                ax1.add_patch(rect3)
                ax1.add_patch(rect4)

            if spec_num in [10]:

                rect1 = patches.Rectangle((5, 1), 1, 7, linewidth=1, edgecolor='b', facecolor='b')
                rect2 = patches.Rectangle((0, 7.8), 1.5, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                rect3 = patches.Rectangle((9.5, 7.8), 3, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                rect4 = patches.Rectangle((4, 7.8), 3, 0.4, linewidth=1, edgecolor='b', facecolor='b')

                rect5 = patches.Rectangle((5, 1 + 16), 1, 7, linewidth=1, edgecolor='b', facecolor='b')
                rect6 = patches.Rectangle((0, 7.8 + 16), 1.5, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                rect7 = patches.Rectangle((9.5, 7.8 + 16), 3, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                rect8 = patches.Rectangle((4, 7.8 + 16), 3, 0.4, linewidth=1, edgecolor='b', facecolor='b')
                
                ax1.add_patch(rect1)
                ax1.add_patch(rect2)
                ax1.add_patch(rect3)
                ax1.add_patch(rect4)

                ax1.add_patch(rect5)
                ax1.add_patch(rect6)
                ax1.add_patch(rect7)
                ax1.add_patch(rect8)

            ax1.set_xlabel('x-coordinates')
            ax1.set_ylabel('y-coordinates')
            ax1.set_title(f'Test Rollout - Combined')
            plt.savefig(f"{imgdir_rollouts}/combined.png")
            plt.close(fig1)

            print('FINISHED PREPARING TEST ROLLOUT IMAGES!!!')


        # TESTING POLICIES
            
        # if testing:
                
        #     print('INITIALIZING TESTING PROCESS...')
            
        #     nn_policies[len(nn_policies)-1] = nn_policies[len(nn_policies)-2]
        #     kappa[len(nn_policies)-1] = kappa[len(nn_policies)-2]
            
        #     prob_list = []
        #     thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        #     counter_dict = {str(threshold): 0 for threshold in thresholds}

        #     for iter in range(200):
        #         system.reset()
        #         env_goal.round_num = iter
        #         abstract_reach.update()
            
        #         kappa_policy = []

        #         for v, edge in enumerate(nn_policies):     
                    
        #             temp = []
        #             for e, edge_policy in enumerate(edge):

        #                 if not isinstance(edge_policy, NNPolicy):
        #                     temp.append(edge_policy)
        #                 elif edge_policy is not None:
        #                     assert (kappa[v][e]!=None)
        #                     kappa_edge = NNPolicy.get_kappa(edge_policy, rnd_no, kappa[v][e])
                
        #                     if rnd_no == 0:
        #                         kappa_edge.mu = edge_policy.mu
        #                         kappa_edge.sigma_inv = edge_policy.sigma_inv

        #                     temp.append(kappa_edge)
        #                 else:
        #                     temp.append(None)
        #             kappa_policy.append(temp)
                  
                        
        #         hierarchical_policy = HierarchicalPolicy(abstract_policy, kappa_policy, abstract_reach.abstract_graph, system.observation_space.shape[0])
                
        #         final_env = ConstrainedEnv(system, abstract_reach, abstract_policy, res_model=None, max_steps=60)
        #         _, prob = print_performance(final_env, hierarchical_policy, stateful_policy=True, save_path=testdir_ar, rnd_no=iter)

        #         prob_list.append(prob)

        #         # Update counts for each threshold
        #         for threshold in thresholds:
        #             if prob >= threshold:
        #                 counter_dict[str(threshold)] += 1

        #         print(counter_dict)

        #         # Break if the last five probabilities are all zero
        #         if len(prob_list) > 5 and all(x == 0.0 for x in prob_list[-5:]):
        #             break

        #     # Save the counter dictionary
        #     with open(f"{testdir}/thresholds.pkl", "wb") as fp:
        #         pickle.dump(counter_dict, fp)

        #     with open(f"{testdir}/iters.pkl", "wb") as fp:
        #         pickle.dump(iter, fp) 

        #     print('TESTING COMPLETE !!!')
        
        # if rollouts_test:

        #     print('PRINTING TEST ROLLOUT IMAGES...')

        #     figure_size = (10, 8)
        #     jump = 1  

        #     fig1, ax1 = plt.subplots(figsize=figure_size)

        #     with open(f"{testdir}/iters.pkl", 'rb') as file1:
        #             iters = pickle.load(file1)

        #     for rnd_no in range(0, iters, jump):
        #         file_path = f'{testdir_ar}/test_rollouts_r{rnd_no}.pkl'

        #         # Open the file in binary read mode
        #         with open(file_path, 'rb') as file:
        #             # Load the object from the file
        #             rollout = pickle.load(file)

        #         data = rollout

        #         points = []
        #         for item in data:
        #             if item == ['separator']:
        #                 break
        #             if isinstance(item, (list, tuple)) and len(item) == 2:
        #                 points.append(item)

        #         if not points:  # Skip if no points
        #             continue

        #         # Separate the x and y coordinates
        #         x_values, y_values = zip(*points)

        #         # Create a new figure for each rollout
        #         fig, ax = plt.subplots(figsize=figure_size)

        #         # Plot the points on the new figure
        #         ax.scatter(x_values, y_values, s=5)  # Adjust the marker size with the 's' parameter
        #         ax.plot(x_values, y_values)

        #         ax1.scatter(x_values, y_values, s=5)  # Adjust the marker size with the 's' parameter
        #         ax1.plot(x_values, y_values)

        #         ax.set_xlabel('x-coordinates')
        #         ax.set_ylabel('y-coordinates')
        #         ax.set_title(f'Test Rollout: Round no {rnd_no}')
        #         plt.savefig(f"{test_imgdir_rollouts}/round_{rnd_no}.png")

        #         # Close the figure to free up memory
        #         plt.close(fig)

        #     ax1.set_xlabel('x-coordinates')
        #     ax1.set_ylabel('y-coordinates')
        #     ax1.set_title(f'Test Rollout - Combined')
        #     plt.savefig(f"{test_imgdir_rollouts}/combined.png")

        #     # Close the figure to free up memory
        #     plt.close(fig1)

        #     print('FINISHED PREPARING TEST ROLLOUT IMAGES!!!')


        


