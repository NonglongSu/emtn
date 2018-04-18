
from sympy import *

#definition of symbolic variables
s1 = Symbol('s1')
s2 = Symbol('s2')
s3 = Symbol('s3')
s4 = Symbol('s4')
s5 = Symbol('s5')
s6 = Symbol('s6')
s7 = Symbol('s7')
s8 = Symbol('s8')
s9 = Symbol('s9')
s10 = Symbol('s10')
s11 = Symbol('s11')
q = Symbol('q')
t = Symbol('t')
beta = Symbol('beta')
alfa_r = Symbol('alfa_r')
alfa_y = Symbol('alfa_y')
pi_a = Symbol('pi_a')
pi_g = Symbol('pi_g')
pi_c = Symbol('pi_c')
pi_t = Symbol('pi_t')
pi_r = Symbol('pi_r')
pi_y = Symbol('pi_y')

full_likelihood = s1*(alfa_r*log(q)/beta + log(q)) + s2*(alfa_y*log(q)/beta+log(q)) \
                    + s3*(log(q)+log(1-exp(alfa_r *(-log(q)/beta)))) \
                    + s4*(log(q)+log(1-exp(alfa_y *(-log(q)/beta)))) + s5*log(1-q) \
                    + s6*log(pi_a) + s7*log(pi_g) + s8*log(pi_c) + s9*log(pi_t) \
                    - s10*log(pi_r) - s11*log(pi_y)
log_likelihood_betaT = diff(full_likelihood,q)
#solve(log_likelihood_betaT,q)
## Error: No algorithms are implemented to solve equation
