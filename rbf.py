#coding: utf-8

import sys
import numpy as np
import numpy.random as random
from numpy import random
from scipy.cluster.vq import kmeans,whiten
from scipy.linalg import norm,pinv,lstsq

class rbf(object):  
    
    def __init__(self, ninput, nout, nk):        

        self.ninput = ninput
        self.nout = nout
        self.nk = nk
        self.b = 0.5
        self.w = random.random((nk, nout))
        self.k = [random.uniform(0,1,ninput) for i in xrange(nk)]
        
    def gauss(self, x, c):
        return np.exp(-self.b * norm(x-c)**2)

    def calcula_vetor_rbf(self, entrada):
        h = np.zeros((len(entrada), self.nk),float)
        
        for ci,c in enumerate(self.k):
            for xi,x in enumerate(entrada):
                h[xi,ci] = self.gauss(c,x)
        #bias
        h = h.tolist()
        for i in xrange(len(h)):
            h[i].insert(0,1)
        
        return np.array(h)

    def train(self, entrada, saida):
        """
        Treina a rede utilizando k-means para 
        determinar os centros das rbfs e utiliza
        o método dos minimos quadrados para calcular
        o vetor de pesos.

        """
        #encontrando os centros com k-means
        wh = whiten(entrada)
        self.k,distortion = kmeans(wh,10)
        #calculando as rbfs
        h = self.calcula_vetor_rbf(entrada)
        #calculando os pesos
        self.w,residual,rank,s = lstsq(h,saida)
        
    def test(self,entrada):

        h = self.calcula_vetor_rbf(entrada)
        f = np.dot(h,self.w)
        s = [norm(i) for i in np.transpose(f)]
        p = s.index(max(s)) 
        q = s.index(min(s))
        s[p] = 1
        s[q] = 0
        
        return f

            
