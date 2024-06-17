import numpy as np
import pandas as pd
from logger import Logger
from abc import ABC

class SelectionStrategy(ABC):
    def select_next_instance():
        pass

class GreedySelectionStrategy(SelectionStrategy):
    def __init__(self, logger : Logger):
        self.logger    =  logger
        self.logPrefix = "Greedy Selection Strategy"

    def select_next_instance(self, acquisition_values : np.array, candidate_instances: pd.DataFrame):
        selected = candidate_instances.iloc[np.argmax(acquisition_values)]["type"] 
        self.logger.info(self.logPrefix, "Greedily Selected: "+ str(selected))
        return selected
                
class RandomSelectionStrategy(SelectionStrategy):
    def __init__(self, logger : Logger):
        self.logger    = logger
        self.logPrefix = "Random Selection Strategy"

    def select_next_instance(self, candidate_instances: pd.DataFrame):
        selected = np.random.choice(candidate_instances, size=1, replace=False)[0]
        self.logger.info(self.logPrefix, "Randomly Selected Zif: " + selected)
        return selected

class SerialSelectionStrategy(SelectionStrategy):
    def __init__(self, logger : Logger):
        self.logger    = logger
        self.logPrefix = "Serial Selection Strategy"

    def select_next_instance(self, candidate_instances: pd.DataFrame, candidate_position: int):
        selected = candidate_instances[candidate_position]
        self.logger.info(self.logPrefix, "Serialy Selected Zif: " + selected)
        return selected

class ProbabilisticSelectionStrategy(SelectionStrategy):

    def __init__(self, logger : Logger):
        self.logger    =  logger
        self.logPrefix = "Probabilistic Selection Strategy"

    def select_next_instance(self, acquisition_values : np.array, candidate_instances: pd.DataFrame):
        
        options = ['max', 'rand']
        probabilities = [0.8, 0.2]
        option = np.random.choice(options, p=probabilities)

        if option == 'max':
            selected = candidate_instances.iloc[np.argmax(acquisition_values)]["type"]
            self.logger.info(self.logPrefix, "Probabilistic selection settled to max value.")
        else:

            candidate_names = candidate_instances["type"].unique()
            selected = np.random.choice(candidate_names, size=1, replace=False)[0]
            self.logger.info(self.logPrefix, "Probabilistic selection settled to random value.")

        self.logger.info(self.logPrefix, "Probabilisticly Selected: "+ str(selected))

        return selected