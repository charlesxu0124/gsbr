from agents.bc import BCAgent
from agents.rlt import RLTAgent
from agents.gsbr import GSBRAgent
from agents.td3bc import TD3BCAgent
from agents.expo import EXPOAgent
from agents.bfn import BFNAgent
from agents.dsrl_na import DSRLNAAgent
from agents.dice_rl import DICERLAgent
from agents.behavior_regularized import (
    BehaviorRegularizedAgent as BehaviorRegularizedAgent,
)
from agents.sac import SACAgent as SACAgent

agents = {
    "sac": SACAgent,
    "bc": BCAgent,
    "rlt": RLTAgent,
    "gsbr": GSBRAgent,
    "td3bc": TD3BCAgent,
    "expo": EXPOAgent,
    "bfn": BFNAgent,
    "dsrl_na": DSRLNAAgent,
    "dice_rl": DICERLAgent,
}
