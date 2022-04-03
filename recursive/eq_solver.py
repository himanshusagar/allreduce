
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
    gammaCoeff = p_1_div_p * n
    betaCoeff = gammaCoeff * 2
    return np.array([alphaCoeff , betaCoeff]);


A = np.array([ getAlphaBeta( 16 , 67108864) ,
               getAlphaBeta( 16 , 16777216) ])

b = np.array([ 49.32134488 , 32.2343047 ])

print(np.linalg.solve(A, b))


