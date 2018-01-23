
# coding: utf-8

# * Les parametres de simulation

# In[1]:


S0=100.
K=110.
T=2
r=0.05
sigma=0.5


# * Calcul de l'esperance:

# In[2]:


import numpy as np
def M1(ss0,rr,tt):
    return ss0*np.exp(rr*tt)


# In[4]:


M1(S0,r,T)


# * Calcul de la variance

# In[5]:


def M2(ss0,rr,tt,ssigma):
    return (ss0**2)*np.exp(2*rr*tt)*(np.exp(ssigma**2*tt)-1)


# In[6]:


M2(S0,r,T,sigma)


# * Fonction de calcul du prix d'un call européen par la méthode de B&S

# In[8]:


def d1(ss0,rr,tt,ssigma,kk):
    return (1./(ssigma*np.sqrt(tt)))*(np.log(ss0/kk)+(r+(0.5)*ssigma**2)*tt)


# In[9]:


def d2(tt,ssigma,dd1):
    return dd1-ssigma*np.sqrt(tt)


# In[10]:


from scipy.stats import norm as nm


# In[11]:


def call_bs(ss0,rr,tt,ssigma,kk):
    dd1=d1(ss0,rr,tt,ssigma,kk)
    
    dd2=d2(tt,ssigma,dd1)

    return ss0*nm.cdf(dd1)-kk*np.exp(-rr*tt)*nm.cdf(dd2)


# In[12]:


call_bs(S0,r,T,sigma,K)


# * Calcul de la moyenne et de la variance :

# In[13]:


N=20000000
import time
start = time.time() 


np.random.seed(0) 
w=np.sqrt(T)*np.random.randn(N)
S=S0*np.exp((r-0.5*sigma*sigma)*T+sigma*w)
print "moyenne = ", S.mean()
print
print "variance = ", S.var()
print 
end = time.time()
print "temps d'execution= ",end - start," secondes"


# * Calcul du prix par la méthode de MonteCarlo

# In[14]:


payoff=np.maximum(S-K*np.ones(N),np.zeros(N))
payoff
cmc=payoff.mean()*np.exp(-r*T)
print "prix par montecarlo = ",cmc


# In[15]:


N=1000000
import time
start = time.time() 


np.random.seed(0) 
w=np.sqrt(T)*np.random.randn(N)
S=S0*np.exp((r-0.5*sigma*sigma)*T+sigma*w)
print "moyenne = ", S.mean()
print
print "variance = ", S.var()
print 
end = time.time()
t1=end - start
var1=S.var()
print "temps d'execution= ",end - start," secondes"


# In[19]:


N=10000
import time
#for N in range(1000,100001,1000):

start = time.time() 


np.random.seed(0) 
w=np.sqrt(T)*np.random.randn(N)
S1=S0*np.exp((r-0.5*sigma*sigma)*T+sigma*w)
S2=S0*np.exp((r-0.5*sigma*sigma)*T-sigma*w)
S_antithetique=0.5*(S1+S2)

end = time.time()

t2=end - start
var2=S_antithetique.var()


payoff1=np.maximum(S1-K*np.ones(N),np.zeros(N))
payoff2=np.maximum(S2-K*np.ones(N),np.zeros(N))
payoff=0.5*(payoff1+payoff2)
cmc=payoff.mean()*np.exp(-r*T)
varianceMonteCarlo=payoff.var()*np.exp(-2*r*T)
ecartype1=payoff.std()*np.exp(-r*T)
print "moyenne = ", S_antithetique.mean()
print
print "variance = ", S_antithetique.var()
print 
print "temps d'execution= ",end - start," secondes"
print
print "Gain en variance = ",int(100*(var2-var1)/var1)," %"
print
print "Penalité en temps de calcul = ",int(100*(t2-t1)/t1)," %"
print
print "prix par montecarlo = ",cmc
print
print "variance par montecarlo = ",varianceMonteCarlo
print
print "ecart type monte carlo = ",ecartype1


# In[23]:


N=1000000
import time
List_Temps=[]
List_Variance=[]
for N in range(1000,100001,1000):

    start = time.time() 


    #np.random.seed(0) 
    w=np.sqrt(T)*np.random.randn(N)
    S1=S0*np.exp((r-0.5*sigma*sigma)*T+sigma*w)
    S2=S0*np.exp((r-0.5*sigma*sigma)*T-sigma*w)
    S_antithetique=0.5*(S1+S2)

    end = time.time()

    t2=end - start
    var2=S_antithetique.var()

    GainVariance= int(100*(var1-var2)/var1)
    List_Temps.append(GainVariance)
    TempCalcul=int(100*(t1-t2)/t1)
    List_Variance.append(TempCalcul)
    
import matplotlib.pyplot as plt

plt.plot(range(1000,100001,1000),List_Temps,"red",label="GainTemps")
plt.plot(range(1000,100001,1000),List_Variance,"blue",label="GainVariance")
plt.show()


# In[39]:


import matplotlib.pyplot as plt
NJours=500
N=10000
T=2
pas=float(T)/float(NJours)
dw=np.sqrt(pas)*np.random.randn(N,NJours)
print dw 
w=np.zeros((N,NJours+1))

w[:,1:]=np.cumsum(dw,axis=1)
S=S0*np.ones((N,NJours+1))
for i in range(NJours+1):
    S[:,i]=S0*np.exp((r-0.5*sigma*sigma)*i*pas+sigma*w[:,i])
for k in range(N):
    plt.plot(range(NJours+1),S[k,:])

plt.show()


# In[41]:


NJours=500
N=10000
T=2
pas=float(T)/float(NJours)
w=np.zeros((N,NJours+1))
w[:,1:]=np.cumsum(dw,axis=1)
S=S0*np.ones((N,NJours+1))
for i in range(NJours+1):
    S[:,i]=S0*np.exp((r-0.5*sigma*sigma)*i*pas+sigma*w[:,i])
Sasian=np.zeros(N)
Sasian=np.sum(S,axis=1)/float(NJours+1)
print Sasian
payoff=np.maximum(Sasian-K*np.ones(N),np.zeros(N))
cmc=payoff.mean()*np.exp(-r*T)
ecartype=payoff.std()*np.exp(-r*T)
print cmc,ecartype


# In[42]:


NJours=50
N=5000
T=2
pas=float(T)/float(NJours)
dw=np.sqrt(pas)*np.random.randn(N,NJours)
w=np.zeros((N,NJours+1))
w[:,1:]=np.cumsum(dw,axis=1)
S=S0*np.ones((N,NJours+1))
for i in range(NJours+1):
    S[:,i]=S0*np.exp((r-0.5*sigma*sigma)*i*pas+sigma*w[:,i])
Sasian=np.zeros(N)
Sasian=np.sum(S,axis=1)/float(NJours+1)
X=np.exp(-r*T)*np.maximum(Sasian-K*np.ones(N),np.zeros(N))
Y=S[:,NJours]

c=-np.cov(X,Y)[0,1]/Y.var()
print(c)


# In[43]:


N=100000
dw=np.sqrt(pas)*np.random.randn(N,NJours)
w=np.zeros((N,NJours+1))
w[:,1:]=np.cumsum(dw,axis=1)
S=S0*np.ones((N,NJours+1))
for i in range(NJours+1):
    S[:,i]=S0*np.exp((r-0.5*sigma*sigma)*i*pas+sigma*w[:,i])
Sasian=np.zeros(N)
Sasian=np.sum(S,axis=1)/float(NJours+1)
X=np.exp(-r*T)*np.maximum(Sasian-K*np.ones(N),np.zeros(N))
print X.mean()," ",X.std()

teta_x=X+c*(-M1(S0,r,T))
print "moyenne = ", teta_x.mean()," ecartype = ",teta_x.std()


# In[44]:


NJours=50
N=5000
T=2
pas=float(T)/float(NJours)
dw=np.sqrt(pas)*np.random.randn(N,NJours)
w=np.zeros((N,NJours+1))
w[:,1:]=np.cumsum(dw,axis=1)
S=S0*np.ones((N,NJours+1))
for i in range(NJours+1):
    S[:,i]=S0*np.exp((r-0.5*sigma*sigma)*i*pas+sigma*w[:,i])
Sasian=np.zeros(N)
Sasian=np.sum(S,axis=1)/float(NJours+1)
X=np.exp(-r*T)*np.maximum(Sasian-K*np.ones(N),np.zeros(N))
Y=np.exp(-r*T)*np.maximum(S[:,NJours]-K*np.ones(N),np.zeros(N))
EY=call_bs(S0,r,T,sigma,K)
c=-np.cov(X,Y)[0,1]/Y.var()
print(c)


# In[45]:


N=100000
dw=np.sqrt(pas)*np.random.randn(N,NJours)
w=np.zeros((N,NJours+1))
w[:,1:]=np.cumsum(dw,axis=1)
S=S0*np.ones((N,NJours+1))
for i in range(NJours+1):
    S[:,i]=S0*np.exp((r-0.5*sigma*sigma)*i*pas+sigma*w[:,i])
Sasian=np.zeros(N)
Sasian=np.sum(S,axis=1)/float(NJours+1)
X=np.exp(-r*T)*np.maximum(Sasian-K*np.ones(N),np.zeros(N))
print X.mean()," ",X.std()
Y=np.exp(-r*T)*np.maximum(S[:,NJours]-K*np.ones(N),np.zeros(N))
EY=call_bs(S0,r,T,sigma,K)
teta_x=X+c*(Y-EY)
print teta_x.mean()," ",teta_x.std()

