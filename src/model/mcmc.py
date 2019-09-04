import numpy as np
import pandas
import pickle
import os
import sklearn
import matplotlib.pyplot as plt


class MCMC():
    def __init__(self, ineteration, walker, sampler, method):
        self.ineteration = ineteration
        self.walker = walker
        self.sampler = sampler
        self.method = method