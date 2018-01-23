
# coding: utf-8

# * Exercice 1 : Simulation d’une variable aléatoire discrète 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

n=10000
U=np.random.rand(n)
#Les valeurs prisent par la v.a.r X
Yi=[1,2,3,4] 

Zi=[0.2,0.5,0.1,0.2]  

#Retourner la variable aléatoire x obéissante a la loi uniforme U[0, 1]
def  Finv (u,Z=Zi,Y=Yi):  
    if u<Z[0]:										
        return Y[0]
    elif u>=Z[0] and u<Z[0]+Z[1]:
        return Y[1]
    elif u>=Z[0]+Z[1] and u<Z[0]+Z[1]+Z[2]:
        return Y[2]
    else :
        return Y[3]				
    
#Transformer U avec la fonction Finv    
XX=map(Finv,U)  
print XX

#Obtenir une échantillon x de v.a i.i.d. qui suit la loi de U.
X = 1 + (U>0.2) + (U>0.7) +(U>0.8) 
print X
plt.hist(X,bins=[0.5,1.5,2.5,3.5,4.5],normed=1,histtype="bar",color="yellow",label="simulation") 
plt.stem([1,2,3,4],[0.2,0.5,0.1,0.2],"red",label="theorie")
plt.legend(loc="best")
plt.show()


# __Les deux diagrammes sont presque identique ce qui nous permet de déduire que l’échantillon est
# bien distribué suivant la loi nu__

# * Exercice 2 : Loi exponentielles et loi connexes

# * Question 1 : Simuler 10000 réalisations de la loi exponentielle $X\hookrightarrow\mathcal{E}(\lambda)$. de paramètre $\lambda$ 

# In[6]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

#Question 1 :
n=10000
lambd=1.
U=np.random.rand(n)
X=-np.log(U)/lambd 
x=np.linspace(0,max(X),1000) 
y=lambd*np.exp(-lambd*x) 

plt.hist(X,bins=int(n**(1/3.)),normed=1,histtype="step",label="simulation")
plt.plot(x,y,"red",label="theorie")
plt.legend(loc="best")
plt.show()


# * Question2: Simuler 10000 réalisations de la loi gamma  $X\hookrightarrow\gamma(n,\lambda)$.

# In[9]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as pllt

nb=10000
n=3
lambd=1.
U=np.random.rand(nb,n)
X=-np.log(U)/lambd
S=np.sum(X,axis=1)

x=np.linspace(0,max(S),1000) 
y=lambd**n/np.prod(range(1,n))*np.exp(-lambd*x)*x**(n-1)
pllt.hist(S,bins=(int(nb**(1/3.))),normed=1,histtype="step",label="simulation")
pllt.plot(x,y,color="red",label="theorie")
pllt.legend(loc="best")
pllt.show()


# * Question 3 Simuler 10000 réalisations de la loi de Poisson $X\hookrightarrow\mathcal{P}(\lambda)$.

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

n=10000
lambd=1

N = np.zeros(n)

for i in range(n):
    m= 0
    s= -np.log(np.random.rand())/lambd
    while s<=1:
    
        s = s - np.log(np.random.rand())/lambd
        m+=1
        
    N[i] = m
    
x = range(10)
y = sps.poisson.pmf(x,lambd)

plt.hist(N,bins=np.arange(-0.5,9.5),normed=1,color="yellow",label="simulation")
plt.stem(x,y,color="red",label="theorie")
plt.legend(loc="best")
plt.show()


# * Exercice 3

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from random import random

n=10000
x=np.sqrt(3)*(2*np.random.rand(n)-1)
S=np.cumsum(x)
plt.plot(range(1,n+1),S/range(1,n+1),label="simulation")
plt.plot([0,n],[0,0],"red",label="esperance")
plt.legend(loc="best")
plt.show()
