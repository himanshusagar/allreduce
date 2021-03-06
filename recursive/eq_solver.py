
from math import log2
import numpy as np

#p = 10 # node count
#n = 1024 #tensor size
#alpha = constant cost to send each message
#beta = cost to send/recv each word of message
#gamma = cost of computation

def getAlphaBeta(p , n):
    p_1_div_p = (p-1)/p
    alphaCoeff = 2 * log2(p)
    #gammaCoeff = p_1_div_p * n
    betaCoeff = p_1_div_p * n * 2
    return np.array([alphaCoeff , betaCoeff]);

def recurse(val , p):
    return log2(p) * val;

A = np.array([ getAlphaBeta( 16 , 33554432) ,
               getAlphaBeta( 16 , 67108864) ])

b = np.array([ recurse( 27.61999477, 16) , recurse( 49.32134488 , 16) ])

print(np.linalg.solve(A, b))


