# params_tn.R
# convert parameters from TN.py to parameters in tn.dawg for comparison
# usage: Rscript params_tn.R true_aY true_aR true_b aY aR b piA piC piG piT

args = as.double(commandArgs(TRUE))

aY = args[1]
aR = args[2]
b = args[3]

aY_em = args[4]
aR_em = args[5]
b_em = args[6]

fr = c(args[7],args[8],args[9],args[10])
m = matrix(c(0,b,aR,b,b,0,b,aY,aR,b,0,b,b,aY,b,0),nrow=4,byrow=TRUE)

m = fr*m
d = sum(t(m)*fr)
d

# define relative error criteria
eps = 0.01

# calculate relative error
e1 = ((aY_em/(fr[2]+fr[4])+b_em)*d - aY)/aY
e2 = ((aR_em/(fr[1]+fr[3])+b_em)*d - aR)/aR
e3 = ((b_em*d) - b)/b

e = c(abs(e1),abs(e2),abs(e3))

if(all(e<eps)) {
    message("Paremeters check passed")
    e
} else {
    message("Parameters check not passed")
    e
}
