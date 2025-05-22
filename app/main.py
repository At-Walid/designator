
import os
import search2_feat_mars as search
import pickle
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
import random 
import numpy as np


GaGan = search.GaGan()
# GaGan.begin_server("./MarsSim/WindowsNoEditor/Mars.exe", 'PhysXCar')




GaGan.searchAlgo(100, 12, 1)