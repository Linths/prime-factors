#!/usr/bin/env python
# coding: utf-8

from Crypto.Util import number
import math
import numpy as np
from collections import namedtuple
from lmgs import *
import cProfile

class GeneratedData:
    def __init__(self, n_bits=256, n_datapoints=100):
        pr = cProfile.Profile()
        pr.enable()

        self.n_bits = n_bits
        self.n_datapoints = n_datapoints

        self.generate_primes()
        
        self.find_firsts()
        self.get_mods()
        self.translate()
        
        self.create_input()
        self.create_output()
        self.create_datapairs()
        
        pr.disable()
        pr.print_stats(sort='time')
    
    def generate_primes(self):
        lst = [(number.getPrime(self.n_bits//2), number.getPrime(self.n_bits//2)) for i in range(self.n_datapoints)]
        self.semiprimes = [(ele[0]*ele[1], min(ele[0], ele[1])) for ele in lst]
    
    
    def is_prime(self, n):
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
    
    def next_prime(self, n):
        if n <= 1:
            return 2
        prime = n
        found = False
        while not found:
            prime = prime + 2
            if self.is_prime(prime):
                found = True
        return prime
    
    def find_firsts(self):
        self.moduli = [3]
        prod = 2*3
        while prod <= 2**self.n_bits:
            next_prime_number = self.next_prime(self.moduli[-1])
            self.moduli.append(next_prime_number)
            prod = prod*next_prime_number
    
    def get_mods(self):
        self.pmod = np.zeros((len(self.semiprimes), len(self.moduli)))
        self.nmod = np.zeros((len(self.semiprimes), len(self.moduli)))
        for i in range(self.pmod.shape[0]):
            for j in range(self.pmod.shape[1]):
                self.pmod[i, j] = self.semiprimes[i][1] % self.moduli[j]
                self.nmod[i, j] = self.semiprimes[i][0] % self.moduli[j]
    
    
    def translate(self):
        self.p_x = np.zeros(self.pmod.shape)
        self.p_y = np.zeros(self.pmod.shape)
        self.n_x = np.zeros(self.nmod.shape)
        self.n_y = np.zeros(self.nmod.shape)
        for i in range(self.p_x.shape[0]):
            for j in range(self.p_x.shape[1]):
                p_temp = 2*math.pi*self.pmod[i, j]/self.moduli[j]
                self.p_x[i, j] = math.cos(p_temp)
                self.p_y[i, j] = math.sin(p_temp)
                n_temp = 2*math.pi*self.nmod[i, j]/self.moduli[j]
                self.n_x[i, j] = math.cos(n_temp)
                self.n_y[i, j] = math.sin(n_temp)
    
    
    def create_input(self):
        self.t_inp = []
        for i in range(self.n_x.shape[0]):
            row_input = [1]
            row_input.extend(self.n_x[i])
            row_input.extend(self.n_y[i])        
            self.t_inp.append(row_input)
    
    
    def create_output(self):
        self.t_outp = []
        for i, m in enumerate(self.moduli):
            t_output = []
            for p in range(len(self.p_x)):
                row_output = [self.p_x[p,i], self.p_y[p,i]]
                t_output.append(row_output)
            self.t_outp.append(t_output)
    
    
    def create_datapairs(self):
        self.datapairs = {}
        for i, m in enumerate(self.moduli):
            ls = []
            for j, ip in enumerate(self.t_inp):
                ls.append(DataPair(ip, self.t_outp[i][j], self.semiprimes[j][0], self.semiprimes[j][1]))
            self.datapairs[m] = ls
    
    
#    def generate_data(self):
            # NIET MEER NODIG NU HET IN DE __INIT__ FUNCTIE STAAT
#        """Main function, which creates the training data with n_primes nr of datapoints 
#        with n_bits nr of bits of the to be factorized semiprimes"""
#        
#        self.generate_primes()
#        self.find_firsts()
#        self.get_mods()
#        
#        self.translate()
#        self.create_input()
#        
#        self.create_output()
#        
#        self.create_datapairs()
#    
#        return self.datapairs


## MANIER VAN AANROEPEN:
# print(GeneratedData().datapairs)
            
            
