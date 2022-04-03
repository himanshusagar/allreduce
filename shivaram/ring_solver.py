
from math import log2
import numpy as np

#p = 10 # node count
#n = 1024 #tensor size
#alpha = constant cost to send each message
#beta = cost to send/recv each word of message
#gamma = cost of computation

def getAlphaBeta(p , n):
    p_1_div_p = (p-1)/p
    alphaCoeff = p * 2
    betaCoeff = p_1_div_p * n * 2
    return np.array([alphaCoeff , betaCoeff]);

def ringReduce(val , p):
    return p * (p-1) * 2 * val;


A = np.array([ getAlphaBeta( 16 , 16777216) ,
               getAlphaBeta( 16 , 33554432) ])

b = np.array([ ringReduce(16.060583114624023, 16) ,
               ringReduce(31.407331466674805 , 16) ])

print(np.linalg.solve(A, b))


