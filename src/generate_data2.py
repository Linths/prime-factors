#!/usr/bin/env python
# coding: utf-8

from Crypto.Util import number
import math
import numpy as np
from collections import namedtuple
from lmgs import *


def generate_primes(n=256, l=100):
    lst = [(number.getPrime(n//2), number.getPrime(n//2)) for i in range(l)]
    sem = [(ele[0]*ele[1], min(ele[0], ele[1])) for ele in lst]
    # print(sem)
    return sem


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(math.sqrt(n) + 1), 6):
        if n % i == 0 or n % (i+2) == 0:
            return False
    return True


def next_prime(n):
    if n <= 1:
        return 2
    prime = n
    found = False
    while not found:
        prime = prime + 2
        if is_prime(prime):
            found = True
    return prime


def find_firsts(n=256):
    first_primes = [3]
    prod = 2*3
    while prod <= 2**n:
        next_prime_number = next_prime(first_primes[-1])
        first_primes.append(next_prime_number)
        prod = prod*next_prime_number
    return first_primes


def get_mods(semiprimes, first_primes):
    p_mods = np.zeros((len(semiprimes), len(first_primes)))
    n_mods = np.zeros((len(semiprimes), len(first_primes)))
    for i in range(p_mods.shape[0]):
        for j in range(p_mods.shape[1]):
            p_mods[i, j] = semiprimes[i][1] % first_primes[j]
            n_mods[i, j] = semiprimes[i][0] % first_primes[j]
    return p_mods, n_mods


def translate(p, n, m):
    p_x = np.zeros(p.shape)
    p_y = np.zeros(p.shape)
    n_x = np.zeros(n.shape)
    n_y = np.zeros(n.shape)
    for i in range(p_x.shape[0]):
        for j in range(p_x.shape[1]):
            p_temp = 2*math.pi*p[i, j]/m[j]
            p_x[i, j] = math.cos(p_temp)
            p_y[i, j] = math.sin(p_temp)
            n_temp = 2*math.pi*n[i, j]/m[j]
            n_x[i, j] = math.cos(n_temp)
            n_y[i, j] = math.sin(n_temp)
    return p_x, p_y, n_x, n_y


def create_input(n_x, n_y):
    t_input = []
    for i in range(n_x.shape[0]):
        row_input = [1]
        row_input.extend(n_x[i])
        row_input.extend(n_y[i])        
        t_input.append(row_input)
    return t_input


def create_output(p_x, p_y, i):
    t_output = []
    for p in range(len(p_x)):
        row_output = [p_x[p,i], p_y[p,i]]
        t_output.append(row_output)
    return t_output


def create_datapairs(inp, outp, moduli):
    dp_dict = {}
    for i, m in enumerate(moduli):
        ls = []
        for j, ip in enumerate(inp):
            ls.append(DataPair(ip, outp[i][j]))
        dp_dict[m] = ls
    return dp_dict


def generate_data(n_bits=256, n_primes=100):
    """Main function, which creates the training data with n_primes nr of datapoints 
    with n_bits nr of bits of the to be factorized semiprimes"""
    
    # print(n_bits)
    # print(n_primes)
    
    semiprimes = generate_primes(n_bits, n_primes)
    moduli = find_firsts(n_bits)
    p, n = get_mods(semiprimes, moduli)
    
    p_x, p_y, n_x, n_y = translate(p, n, moduli)
    t_inp = create_input(n_x, n_y)
    
    t_outp = []
    for i, m in enumerate(moduli):
        outp = create_output(p_x, p_y, i)
        t_outp.append(outp)
    
    datapairs = create_datapairs(t_inp, t_outp, moduli)

    return datapairs
