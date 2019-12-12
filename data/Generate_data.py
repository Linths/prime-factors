#!/usr/bin/env python
# coding: utf-8

# In[151]:


from Crypto.Util import number
import random
import math
import numpy as np
from collections import namedtuple


# In[152]:


def generate_primes(n=256, l=100):
    lst = [(number.getPrime(n//2),number.getPrime(n//2)) for i in range(l)]
    sem = [(ele[0]*ele[1], min(ele[0],ele[1])) for ele in lst]
    return sem


# In[153]:


def isPrime(n):
    if (n<=1): return False
    if (n<=3): return True
    if (n%2==0 or n%3==0):
        return False
    for i in range(5,int(math.sqrt(n)+1),6):
        if (n%i ==0 or n%(i+2)==0):
            return False
    return True

def nextPrime(N):
    if (N<=1): return 2
    prime = N
    found = False
    while (not found):
        prime = prime + 2
        if (isPrime(prime) == True):
            found = True
    return prime


# In[154]:


def find_firsts(n=256):
    first_primes = [2,3]
    prod = 2*3
    while (prod <= 2**n):
        next_prime = nextPrime(first_primes[-1])
        first_primes.append(next_prime)
        prod = prod*next_prime
    return first_primes


# In[155]:


def get_mods(semiprimes, first_primes):
    p_mods = np.zeros((len(semiprimes),len(first_primes)))
    n_mods = np.zeros((len(semiprimes),len(first_primes)))
    for i in range(p_mods.shape[0]):
        for j in range(p_mods.shape[1]):
            p_mods[i,j] = semiprimes[i][1]%first_primes[j]
            n_mods[i,j] = semiprimes[i][0]%first_primes[j]
    return p_mods, n_mods


# In[156]:


def translate(p, n, m):
    p_x = np.zeros(p.shape)
    p_y = np.zeros(p.shape)
    n_x = np.zeros(n.shape)
    n_y = np.zeros(n.shape)
    for i in range(p_x.shape[0]):
            for j in range(p_x.shape[1]):
                p_x[i,j] = math.cos(2*math.pi*p[i,j]/m[j])
                p_y[i,j] = math.sin(2*math.pi*p[i,j]/m[j])
                n_x[i,j] = math.cos(2*math.pi*n[i,j]/m[j])
                n_y[i,j] = math.sin(2*math.pi*n[i,j]/m[j])
    return p_x, p_y, n_x, n_y


# In[157]:


def create_ntuple(p, n, m):
    p_x,p_y,n_x,n_y = translate(p, n, m)
    t_input = []
    t_output = []
    for i in range(p_x.shape[0]):
        row_input = [1]
        row_input.extend(n_x[i])
        row_input.extend(n_y[i])
        row_output = [1]
        row_output.extend(p_x[i])
        row_output.extend(p_y[i])
        
        t_input.append(row_input)
        t_output.append(row_output)
    
    TrainData = namedtuple('train_v1',['input','output'])
    return TrainData(t_input,t_output)


# In[158]:


def run_all():
    n_bits = 256
    n_primes = 100
    
    semiprimes = generate_primes(n_bits, n_primes)
    moduli = find_firsts()
    
    p, n = get_mods(semiprimes, moduli)
    
    return create_ntuple(p, n, moduli)


# In[159]:


traindata = run_all()


# In[160]:


traindata


# In[ ]:




