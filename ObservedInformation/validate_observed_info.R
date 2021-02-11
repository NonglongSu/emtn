#

t.hat = 0.20007508530163765
aR.hat = 0.4769256422510068
aY.hat = 0.7258680923376556
b.hat = 0.9611676228358045
pi.hat = c(0.3000642043733854, 0.2000018110159562, 0.20023046164754882, 0.29970352296310954,0,0)
pi.hat[5] = pi.hat[1] + pi.hat[3]
pi.hat[6] = pi.hat[2] + pi.hat[4]


rho.hat = aR.hat/aY.hat
Ts.hat = 2.0*aR.hat*pi.hat[1]*pi.hat[3]/pi.hat[5] + 2.0*aY.hat*pi.hat[2]*pi.hat[4]/pi.hat[6] + 2.0*b.hat*(pi.hat[1]*pi.hat[3] + pi.hat[2]*pi.hat[4])
Tv.hat = 2.0*b.hat*pi.hat[5]*pi.hat[6]
R.hat = Ts.hat/Tv.hat

################################################################################
t = 0.2
aY = 0.15/0.208
aR =0.1/0.208
b = 0.2/0.208
pi = c(0.3,0.2,0.2,0.3,0.5,0.5)

rho = aR/aY
Ts = 2.0*aR*pi[1]*pi[3]/pi[5] + 2.0*aY*pi[2]*pi[4]/pi[6] + 2.0*b*(pi[1]*pi[3] + pi[2]*pi[4])
Tv = 2.0*b*pi[5]*pi[6]
R = Ts/Tv

################################################################################

theta.hat = c(t.hat,R.hat,rho.hat)
theta = c(t,R,rho)

theta1.hat = c(aR.hat,aY.hat,b.hat)
theta1 = c(aR,aY,b)

I = matrix(0,3,3)
I[1,1] = 1.47622157e+12
I[1,2] = 4.33680869e-19
I[1,3] = 4.33680869e-19
I[2,1] = 4.33680869e-19
I[2,2] = 1.44691796e+12
I[2,3] = -2.16840434e-19
I[3,1] = 4.33680869e-19
I[3,2] = -2.16840434e-19
I[3,3] = 7.78758014e+12

theta_diff = matrix(theta.hat-theta)
x.square = t(theta_diff) %*% I %*% theta_diff
pchisq(x.square,3,lower.tail=FALSE)

theta1_diff = matrix(theta1.hat-theta1)
x1.square = t(theta1_diff) %*% I %*% theta1_diff
pchisq(x1.square,3,lower.tail = FALSE)
