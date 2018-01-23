
# coding: utf-8

# In[2]:


from scipy.stats import norm as nm
import numpy as np


# In[3]:


S0=100.
K=140.
T=2.
r=0.05
CM=60.545


# Première étape Ecrire la fonction du prix exact d’un Call europeen par B&S

# * Calcul de D1 par la Fonction : 

# In[6]:


def d1(ss0,rr,tt,ssigma,kk):
    return (1./(ssigma*np.sqrt(tt)))*(np.log(ss0/kk)+(r+(0.5)*ssigma**2)*tt)


# * Calcul de D2 par la Fonction :

# In[7]:


def d2(tt,ssigma,dd1):
    return dd1-ssigma*np.sqrt(tt)


# * On Applique B&S

# In[8]:


def call_bs(ss0,rr,tt,ssigma,kk):
    dd1=d1(ss0,rr,tt,ssigma,kk)
    
    dd2=d2(tt,ssigma,dd1)

    return ss0*nm.cdf(dd1)-kk*np.exp(-rr*tt)*nm.cdf(dd2)


# * La methode de dichotomie :
# On choisit aléatoirement des valeurs du sigma afin de rapprocher le résultat du CM

# In[9]:


print call_bs(S0,r,T,1.31632,K)


# In[10]:


print call_bs(S0,r,T,1.5,K)


# * Le problème pour résoudre l’équation ci-dessus est qu’elle fait intervenir des intégrales. Il faut donc avoir recours à des procédures numérique pour en approximer la solution.
# * Une méthode intuitive serait de commencer par une volatilité très élevée, puis de baisser progressivement pour se rapprocher de la bonne valeur, en tâtonnant
# * la volatilite implicite par Methode de Dichotomie

# * diviser un intervalle par deux a chaque etape et garder le bon intervalle

# In[12]:


def f(ss0,rr,tt,ssigma,kk,cm):
    
    return call_bs(ss0,rr,tt,ssigma,kk)-cm 


# In[17]:


import time
def dichotomie(S0,K,r,T,cm,sig_Inf=0.05,sig_Sup=3.0,tol=0.001,lim_cpt=100):
    start = time.time() 
    cpt=0
    
    if (f(S0,r,T,sig_Inf,K,cm)*f(S0,r,T,sig_Sup,K,cm)) <0 :
        
        Sig_Milieu=(sig_Inf+sig_Sup)/2
      
        while ((np.abs(f(S0,r,T,Sig_Milieu,K,cm))>tol)and (cpt<lim_cpt)) :
            
        
            if (f(S0,r,T,Sig_Milieu,K,cm))<0:
                sig_Inf=Sig_Milieu
            else:
                sig_Sup=Sig_Milieu
            
            Sig_Milieu=(sig_Inf+sig_Sup)/2.0
            cpt+=1
            
        end = time.time()
        print "temps d'execution= ",end - start," secondes"
        return (Sig_Milieu,cpt)       
    else:
        return (0,cpt)


# * après avoir implementer la fonction nous allons voir le resultat de l'execution :

# In[18]:


print dichotomie(S0,K,r,T,CM)


# * Nbre d itérations : 13
# 
# Ce résultat dépend principalement de la tolérance et de l'intervalle choisi

# * on refait l'essai en changeant la borne supérieure

# * temps d'execution=  0.00399994850159  secondes
# (1.3163150787353515, 16)
# * Nbre d itérations : 16

# * Cependant, il est possible d’améliorer cette idée en utilisant l’algorithme de Newton-Raphson. L’idée est que grâce à la formule de Taylor pour une fonction f dérivable au moins une fois: 
# * f(x1) ≈ f(x0) + f0(x0)(x1 − x0)
# et on cherche à rrésoudre f(x1) = 0, ie 
# * x1 = x0 −f(x0)/f0(x0)
# * Graphiquement, x1 est le point qui est à l’intersection entre la tangente de la courbe au point x0 et l’axe des abscisses. On voit alors que l’on se rapproche de la solution en suivant la direction de la courbe. Il reste alors à itérer le processus grâce à la formule de récurrence
# * xn+1 = xn −f(xn)f0(xn)
# * Très vite, xn converge vers la solution

# In[19]:


def Nprim(X):
    return (1./np.sqrt(2*np.pi))*np.exp(-0.5*X*X)


# In[20]:


def Fprim(SS0,tt,rr,KK,sig):
    return SS0*np.sqrt(tt)*Nprim(d1(SS0,rr,tt,sig,KK))


# In[21]:


import time

def Newton(SS0,KK,rr,tt,cm,sigIn=1.0,tol=0.001):
    start = time.time() 
    sig = sigIn 
    cpt = 0
    while (np.abs( f(SS0,rr,tt,sig,KK,cm) ) > tol ):
        sig = sig - f(SS0,rr,tt,sig,KK,cm) / Fprim(SS0,tt,rr,KK,sig)
        #print sig
        
        cpt+=1
    print "Le nombre d'iteration est :" ,cpt
    end = time.time()
    print "temps d'execution= ",end - start," secondes"
    return sig


# In[22]:


print Newton(S0,K,r,T,CM)


# * la méthode de NewtonRaphson est plus efficace que celle de Dichotomie 
# * nombre d'iteration = 3
# * temps d'execution = 0.0039

# On va essayer de proceder par la methode de MonteCarlo

# In[23]:


def call_bs_MonteCarlo(S0,r,T,sigma,K,N=20000000):
    
    np.random.seed(0) 
    w=np.sqrt(T)*np.random.randn(N)
    S=S0*np.exp((r-0.5*sigma*sigma)*T+sigma*w)

    payoff=np.maximum(S-K*np.ones(N),np.zeros(N))
    cmc=payoff.mean()*np.exp(-r*T)
    return cmc


# * On va calculer une valeur approchée de sigma :

# In[24]:


call_bs_MonteCarlo(S0,r,T,1.316303,K)


# * la volatilité implicite par Dichotomie en utilisant MonteCarlo

# In[25]:


def f_motecarlo(ss0,rr,tt,ssigma,kk,cm):
    
    return call_bs_MonteCarlo(S0,r,T,ssigma,K)-cm 


# In[26]:


import time
def dichotomie(S0,K,r,T,cm,sig_Inf=0.05,sig_Sup=3.0,tol=0.001,lim_cpt=100):
    start = time.time() 
    cpt=0
    
    if (f_motecarlo(S0,r,T,sig_Inf,K,cm)*f_motecarlo(S0,r,T,sig_Sup,K,cm)) <0 :
        
        Sig_Milieu=(sig_Inf+sig_Sup)/2
       
        while ((np.abs(f_motecarlo(S0,r,T,Sig_Milieu,K,cm))>tol)and (cpt<lim_cpt)) :
            
        
            if (f_motecarlo(S0,r,T,Sig_Milieu,K,cm))<0:
                sig_Inf=Sig_Milieu
            else:
                sig_Sup=Sig_Milieu
            
            Sig_Milieu=(sig_Inf+sig_Sup)/2.0
            cpt+=1
            
        end = time.time()
        print "temps d'execution= ",end - start," secondes"
        return (Sig_Milieu,cpt)       
    else:
        return (0,cpt)


# In[27]:


print dichotomie(S0,K,r,T,CM)


# * Nombre d'itération:  15 
# * Temps d'execution : 69.203 secondes
# 

# * La methode de Monte Carlo demande beaucoup de ressources Hard

# *  la volatilité implicite par NewtonRaphson en utilisant MonteCarlo

# In[28]:


def f_motecarlo(ss0,rr,tt,ssigma,kk,cm):
    
    return call_bs_MonteCarlo(S0,r,T,ssigma,K)-cm 


# In[29]:


import time

def Newton(SS0,KK,rr,tt,cm,sigIn=1.0,tol=0.001):
    start = time.time() 
    sig = sigIn 
    cpt = 0
    while (np.abs( f_motecarlo(SS0,rr,tt,sig,KK,cm) ) > tol ):
        sig = sig - f_motecarlo(SS0,rr,tt,sig,KK,cm) / Fprim(SS0,tt,rr,KK,sig)
        cpt+=1
    print "Le nombre d'iteration est :" ,cpt
    end = time.time()
    print "temps d'execution= ",end - start," secondes"
    return sig


# In[30]:


print Newton(S0,K,r,T,CM)


# * Le resultat est obtenu :
# * temps d'execution = 14.46 secondes
# * nbre d'iterations = 3

# * le résultat est meilleur que celui obtenu par la méthode de dichotomie mais c'est encore loin de la formule exacte de B&S
# 
# * le retard de MonteCarlo est considerable 
