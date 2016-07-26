# Self - tracking data prep

#reset
import csv
import numpy as np
import pandas as pd
import scipy as scipy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#set random seed

np.random.seed(123)

# the basics

#make fake obs
#true A
A_true=np.array([[0.95, 0.05],[0.05, 0.95]], dtype=float)

#true pi
pi_true=np.array([0.95, 0.05], dtype=float)

#true mu & sigma
#mu_t=np.array([[100, 120 ],[-300, -350]],dtype=float)
mu_t=np.array([[50, 55 ],[-50, -55]],dtype=float)
#sigma_t=np.array([[0.4,0,0,0.2],[10,0,0,12]],dtype=float)
sigma_t=np.array([[10.0,0,0,10.0],[10,0,0,12]],dtype=float)

    
#number of possible states
N=2

#number of variables 
M=2

#number of time
T=50


#generate data
x=np.zeros(shape=(T,M))
z_t=np.zeros(T)
for t in range(T):
    if (t==0):
        #draw z0 from categorical with prob pi_true
        z_t[t]=np.random.choice(range(N),1,p=pi_true)   
    else:
        #draw z_t from p(z_t/z_t-1, A)
        z_t[t]=np.random.choice(range(N),1,p=A_true[z_t[t-1],:])

    # set x from multi normal given mu_k[z] and sigma_k[z]
    x[t]=np.random.multivariate_normal(mu_t[z_t[t]],sigma_t[z_t[t],:].reshape((2,2)))


#model learning

# pi: 
log_pi=np.zeros(shape=(N))
log_pi.fill(np.log(1./N))

#initialize values for model parameters
# A: transition prob NxN
log_A=np.zeros(shape=(N,N))
log_A.fill(np.log(1./N))

# initialize mu and sigma
# 2 possible states

mu=np.array([[100,120],[-100,-100]], dtype=float)

sigma=np.array([[10.8,0,0,10.6],[10,0,0,10.1]], dtype=float)


max_iter=1000
iter=0
log_p=0
log_p_old=0

while (log_p>=log_p_old and iter<max_iter):
    iter+=1
    log_p_old=log_p
    
##################
#1) Forward Algo.#
##################
#(alpha = prob of the partial obs seq up to t)
    log_alpha=np.zeros(shape=(T,N))
    
#t=0: log(alpha_0(i))=log(pi*b_ti)
    for i in range(N): 
        log_b_ti= multivariate_normal.logpdf(x[0,:], mu[i],sigma[i,:].reshape((2,2)))
        log_alpha[0,i]=log_pi[i]+log_b_ti

#compute alpha_1(i) to alpha_N-1(i)
#state j at time t
#state i at time t-1
    for t in range(1,T):
        for j in range(N):
            k=np.zeros(N)
            log_b_tj= multivariate_normal.logpdf(x[t,:], mu[j],sigma[j,:].reshape((2,2)))
            for i in range(N):
                k[i]=log_alpha[t-1,i]+log_A[i,j]
            max_k=np.max(k)
            log_alpha[t,j]=log_b_tj+max_k+np.log(np.sum(np.exp(k-max_k)))   
####################
# #backward algo. ##       
####################
#start from the end

    log_beta=np.zeros(shape=(T,N))
# end of sequence with 100% prob    
    #log_beta[T-1,i]=[np.log(1), np.log(1)]=[0.,0.]

# beta T-2, ...,0
#j = state at time t
#f = state at time t+1
    for t in range(T-2,-1,-1):
        for j in range(N):
            k=np.zeros(N)
            for f in range(N):
                log_b_tf=multivariate_normal.logpdf(x[t+1,:], mu[f],sigma[f,:].reshape((2,2)))
                k[f]=log_A[j,f]+log_b_tf+log_beta[t+1,f]
            max_k=np.max(k)
            log_beta[t,j]=max_k+np.log(np.sum(np.exp(k-max_k)))   
##########        
##gamma###
##########            
  
#p(obs\model)=denom (scalar)
#p(x)=sum alpha_T(i) over all i
#=log_p(x)=log(sum(alpha_t[i]))

    k=np.zeros(N)
    k=log_alpha[T-1,:]
    max_k=np.max(k)
    log_p_X=max_k+np.log(np.sum(np.exp(k-max_k)))

    
#gamma_ij

    log_gamma_ij=np.zeros(shape=(T-1,N,N))
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                numer=0
                log_b_tj=multivariate_normal.logpdf(x[t+1,:], mu[j],sigma[j,:].reshape((2,2)))
                numer=log_alpha[t,i]+log_A[i,j]+log_b_tj+log_beta[t+1,j]
                log_gamma_ij[t,i,j]=numer-log_p_X               

     
#log_gamma_i

    log_gamma_i=np.zeros(shape=(T,N))

    for t in range(T-1):
        for i in range(N):
            k=np.zeros(N)
            k=log_gamma_ij[t,i,:]
            max_k=np.max(k)
            log_gamma_i[t,i]=max_k+np.log(np.sum(np.exp(k-max_k)))
                
#log_gamma[T-1,i] (since there is no next state)
    log_gamma_i[T-1,:]=log_alpha[T-1,:]-log_p_X


#log_gamma_i v2
#log_gamma_i2[t,i]

#gamma_t(i)=alpha_t(i)*beta_t(i)/p(x)
#log_gamma_t(i)=log(alpha[t,i]*beta[t,i])-log_p(x)
    log_gamma_i2=np.zeros(shape=(T,N))
    for t in range(T):
        for i in range(N):
            log_gamma_i2[t,i]=log_alpha[t,i]+log_beta[t,i]-log_p_X

#M-STEP

####################
#UPDATE PARAMETERS #
###################
#pi
    log_pi=log_gamma_i[0,:]


#log_A[i,j]:
    for i in range(N):
        k_denom=np.zeros(T-2)
        k_denom=log_gamma_i[0:T-2,i]
        max_k_denom=np.max(k_denom)

        for j in range(N):
            k_numer=np.zeros(T-2)
            k_numer=log_gamma_ij[:,i,j]
            max_k_numer=np.max(k_numer)

            log_A[i,j]=(max_k_numer+np.log(np.sum(np.exp(k_numer-max_k_numer)))-max_k_denom-np.log(np.sum(np.exp(k_denom-max_k_denom))) )

#mu

    for i in range(N):
        numer=0
        denom=0
        for t in range(T):
            gamma_i=np.exp(log_gamma_i[t,i]-np.max(log_gamma_i[:,i]))
            numer+=gamma_i*x[t,:]
            denom+=gamma_i
        mu[i]=numer/np.float(denom)
            

#sigma
    sigma=np.zeros(shape=(N,4)) 
    for i in range(N):
        numer=0
        denom=0
        for t in range(T):
            x_diff=np.matrix(x[t,:]-mu[i,:]).T
            x_dot=x_diff.dot(x_diff.T)
            gamma_i=np.exp(log_gamma_i[t,i]-np.max(log_gamma_i[t,:]))
            numer+=gamma_i*x_dot
            denom+=gamma_i
        s=numer/denom
        sigma[i,:]=s.flatten()

        
#update log_p 
    log_p=-log_p_X


###############
#viterbi algo##
###############
v=np.zeros(shape=(N,T))
ptr=np.zeros(shape=(N,T))

for t in range(T):
    for j in range(N):
        temp=np.zeros(N)
        for i in range(N):
            if (t==0):
                log_b_tj=multivariate_normal.logpdf(x[t,:], mu[j],sigma[j,:].reshape((2,2)))
                temp[i]=log_pi[i]+log_b_tj
            else:
                log_b_tj=multivariate_normal.logpdf(x[t,:], mu[j],sigma[j,:].reshape((2,2)))
                temp[i]=v[i,t-1]+log_A[i,j]+log_b_tj
        v[j,t]=np.max(temp)
        ptr[j,t]=np.argmax(temp)



#Traceback

z=np.zeros(T)
z.fill(float('-inf'))
#where to start trace back?
z[T-1]=np.argmax(v[:,T-1])
#from T-2 to 0:
for t in range(T-1,0,-1):
    z[t-1]=ptr[z[t],t]



#plots
plt.plot(x[:,0],x[:,1])
plt.scatter(x[:,0],x[:,1], c=z, s=100)
plt.title('Simulated Data Sequence with State Classification')
plt.show()


#contour map

#cluster0
s0_1=sigma[0,:].reshape((2,2))[0,0]
s0_2=sigma[0,:].reshape((2,2))[1,1]
s0_12=sigma[0,:].reshape((2,2))[0,1]

m0_1=mu[0,0]
m0_2=mu[0,1]

#cluster1
s1_1=sigma[1,:].reshape((2,2))[0,0]
s1_2=sigma[1,:].reshape((2,2))[1,1]
s1_12=sigma[1,:].reshape((2,2))[0,1]

m1_1=mu[1,0]
m1_2=mu[1,1]


#plots
delta = 1.0
p = np.arange(-100.0, 100.0, delta)
y = np.arange(-100.0, 100.0, delta)
P, Y = np.meshgrid(y, p)
Z1 = mlab.bivariate_normal(P, Y, s0_1, s0_2, m0_1, m0_2, s0_12)
Z2 = mlab.bivariate_normal(P, Y, s1_1, s1_2, m1_1, m1_2, s1_12)

plt.figure()
plt.contour(P, Y, Z1, colors='r')   
plt.contour(P, Y, Z2, colors='g')
plt.title('Probability Density Distribution associated with Hidden States')
plt.show()                



