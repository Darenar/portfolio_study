from typing import List, Iterable
import logging

import pandas as pd
from scipy.stats import norm


class LogMixin(object):
    @property
    def logger(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)
    

class CentroidSort:
    
    @staticmethod
    def get_centroid(n_elements: int) -> List[float]:
        """
        Generate a sort vector from the paper R. Almgren and N. Chriss, "Portfolios from Sorts" 
    
        Parameters
        ----------
        n_elements : int
            Number of elements in the sort

        Returns
        -------
        List[float]
            List with centroid values
        """
        alpha = 0.4424  - 0.1185 * (n_elements ** (-0.21))
        return [norm.ppf(
            (n_elements + 1 - i - alpha) / (n_elements - (2 * alpha) + 1)
        ) for i in range(1, n_elements+1)]
    
    def get_ranking(self, order: Iterable[float]):    
        return self.scale_range(self.get_centroid(len(order)))
    
    @staticmethod
    def scale_range(centroid: List[float]) -> List[float]:
        min_centroid_val = min(centroid)
        max_centroid_val = max(centroid)
        print(max_centroid_val - min_centroid_val)
        return [
            ((c - min_centroid_val) * 0.1) / (max_centroid_val - min_centroid_val) - 0.05
            for c in centroid
        ]
