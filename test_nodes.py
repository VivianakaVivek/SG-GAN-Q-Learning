import sys, os, time, random
import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from base_paper.network_env import NetworkEnvironment, Generator
from base_paper.routing import MultiAgentRouter as QLRouter
from my_research.pi_ddqn_routing import MultiAgentPIDDQNRouter as PIDDQNRouter
from config import BETA_WEIGHT, SCALE_TRAIN_EPOCHS_QL, SCALE_TRAIN_EPOCHS_PIDDQN, GAN_NOISE_DIM

nds = [35, 41]
evs = [50]

import pickle
with open('shared_map.pkl', 'rb') as f:
    M = pickle.load(f)

env = NetworkEnvironment()
env.real_edges = M['env_real_edges']
env.real_coords = M['env_real_coords']
env.x_min, env.x_max = M['env_x_min'], M['env_x_max']
env.y_min, env.y_max = M['env_y_min'], M['env_y_max']

print("Nodes | QL Energy | QL Time | PI-DDQN Energy | PI-DDQN Time")
print("-" * 65)

for nd in nds:
    gen = Generator(GAN_NOISE_DIM, nd*2)
    G, _ = env.synthesize_graph(gen, nd, max(1,int(nd*0.24)), connection_threshold=0.25)
    
    # Train QL
    ql_router = QLRouter(G, 50, BETA_WEIGHT)
    ql_router.train(SCALE_TRAIN_EPOCHS_QL)
    ql_e, ql_t = ql_router.evaluate(2)
    
    # Train PIDDQN
    piddqn_router = PIDDQNRouter(G, 50, BETA_WEIGHT)
    piddqn_router.train(SCALE_TRAIN_EPOCHS_PIDDQN)
    pi_e, pi_t = piddqn_router.evaluate(2)
    
    print(f"{nd:5d} | {ql_e:9.2f} | {ql_t:7.1f} | {pi_e:14.2f} | {pi_t:12.1f}")
