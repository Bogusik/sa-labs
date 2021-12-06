import numpy as np
from scipy import special
from src.model.basis_gen import transform_to_standard
class Model:
    def __init__(self, data, degrees, polynom_type,dimensions, split_lambda ):
        self.data =data
        self.degrees = degrees
        self.polynom_type = polynom_type
        self.dimensions = dimensions
        self.split_lambda = split_lambda
    def __choose_poly_type(self):
        if self.polynom_type =='chebyshev':
            self.polynom = special.eval_sh_chebyt
        elif self.polynom_type =='chebyshev_2_type':
            self.polynom = special.eval_sh_chebyu
        elif self.polynom_type == 'legandre':
            self.polynom = special.eval_sh_legendre
        elif self.polynom_type == 'laguerre':
            self.polynom = special.eval_laguerre
        elif self.polynom_type == 'hermite':
            self.polynom = special.eval_hermite
    def __normalize_data(self):
        self.norm_data = (self.data-self.data.min()) / \
            (self.data.max()-self.data.min())
    def __split_data(self):
        X1 = self.norm_data.iloc[:, 0:self.dimensions[0]].to_numpy()
        X2 = self.norm_data.iloc[:, self.dimensions[0]
            :self.dimensions[1]+self.dimensions[0]].to_numpy()
        X3 = self.norm_data.iloc[:, self.dimensions[1]+self.dimensions[0]
            :self.dimensions[2]+self.dimensions[1]+self.dimensions[0]].to_numpy()
        Y = self.norm_data.iloc[:, self.dimensions[2] +
                           self.dimensions[1]+self.dimensions[0]:].to_numpy()
        self.X = [X1,X2,X3]
        self.Y = Y
    def __build_phi(self):
        phi = np.ndarray(shape=(self.Y.shape[0], 0), dtype=float)
        for i in range(len(self.X)):
            for j in range(self.X[i].shape[1]):
                for k in range(self.degrees[i]+1):
                    phi_i = np.array([self.polynom(k, self.X[i][n, j]) for n in range(self.X[i].shape[0]) ]).reshape(self.Y.shape[0],1)
                    phi = np.concatenate((phi,phi_i) ,axis = 1)
        self.phi = phi

 

    def __calculate_lambda(self):
        lambd = np.ndarray(shape = (self.phi.shape[1],0), dtype = float)
        for i in range(self.Y.shape[1]):
            lambd = np.append(lambd, np.linalg.lstsq(self.phi,self.Y[:,i])[0].reshape(self.phi.shape[1],1), axis=1)
        self.lambd = lambd
 

    def __build_psi(self):
        def build_psi_i(lambd):
            '''
            return matrix xi1 for b1 as matrix
            :param A:
            :param lamb:
            :param p:
            :return: matrix psi, for each Y
            '''
            psi = np.ndarray(shape=(self.phi.shape[0], 7))
            j = 0 #iterator in lamb and A
            l = 0 #iterator in columns psi
            for k in range(len(self.X)): # choose X1 or X2 or X3
                for s in range(self.X[k].shape[1]):# choose X11 or X12 or X13
                    for i in range(self.X[k].shape[0]):
                        psi[i,l] = self.phi[i,j:j+self.degrees[k]].dot(lambd[j:j+self.degrees[k]])
                    j+=self.degrees[k]
                    l+=1  
            return psi

        self.psi =[]
        for i in range(self.Y.shape[1]):
            self.psi.append(build_psi_i(self.lambd[:,i]))
      

    def __calculate_a(self):
        self.a = np.ndarray(shape=( 7,0), dtype=float)
        for i in range(self.Y.shape[1]):
            temp = np.linalg.lstsq(self.psi[i], self.Y[:,i],rcond=None )[0]
             
            self.a = np.append(self.a, temp.reshape(temp.shape[0],1), axis=1)
         

    def __build_Fi(self):
        def built_Fi(psi, a):
            '''
            not use; it used in next function
            :param psi: matrix psi (only one
            :param a: vector with shape = (6,1)
            :param degf:  = [3,4,6]//fibonachi of deg
            :return: matrix of (three) components with F1 F2 and F3
            '''
            deg = [self.dimensions[0],self.dimensions[0]+self.dimensions[1],self.dimensions[0]+self.dimensions[1]+self.dimensions[2]]
            # deg = [2,4,7]#обобщить
            m = len(self.X) # m  = 3
            F1i = np.ndarray(shape = (self.X[0].shape[0],m),dtype = float)
            k = 0 #point of begining columnt to multipy
            for j in range(m): # 0 - 2
                for i in range(self.X[0].shape[0]): # 0 - 49
                    F1i[i,j] = psi[i,k:deg[j]].dot(a[k:deg[j]])
                k = deg[j]
            return np.array(F1i)

        self.Fi=[]
        for i in range(self.Y.shape[1]):
            self.Fi.append(built_Fi(self.psi[i],self.a[:,i]))

    def __calculate_c(self):
        self.c = np.ndarray(shape = (len(self.X),0),dtype = float)
        for i in range(self.Y.shape[1]):
            
            self.c = np.append(self.c,\
                 np.linalg.lstsq(self.Fi[i], self.Y[:,i])[0].reshape(len(self.X),1),axis = 1)

    def __build_F(self):
        F = np.ndarray(self.Y.shape, dtype = float)
        for i in range(len(self.X)):
            for j in range(F.shape[1]):#2
                for i in range(F.shape[0]): #50
                    F[i,j] = self.Fi[j][i,:].dot(self.c[:,j])
        self.F = F

    def build(self):
        self.__normalize_data()
        self.__choose_poly_type()
        self.__split_data()
        self.__build_phi()
        self.__calculate_lambda()
        self.__build_psi()
        self.__calculate_a()
        self.__build_Fi()
        self.__calculate_c()
        self.__build_F()
    def get_norm_error(self):
        return np.abs(self.Y -self.F)
    def get_metric(self):
        return self.get_norm_error().max(axis=0)


class ResultPrinter(object):
    def __init__(self,model):
       self.c = model.c
       self.poly_type = model.polynom_type
       self.a = model.a
       self.phi = model.phi
       self.Y = model.Y
       self.dimensions = model.dimensions
       self.degrees = list(map(lambda x:x+1,model.degrees))
       self.lambd = model.lambd

    def form_Psi_i(self,yi):
        iter_lambda = iter(self.lambd[:,yi].tolist())
        strings = []
        for i in range(len(self.dimensions)):
            for j in range(self.dimensions[i]):
                temp ="Psi{i}{j} = ".format(i =i+1,j=j+1)
                for k in range(self.degrees[i]):
                    lambd = next(iter_lambda)
                    temp += '{0:+}'.format(lambd) + "T(x)^{k}".format(k=k)
                strings.append(temp)
        return strings 
    
    def form_Psi(self):
        results = {}
        for i in range(self.Y.shape[1]):
            results["For Y{i}".format(i=i+1)] = self.form_Psi_i(i)
        return results

    def form_Fi_i(self,yi):
        iter_lambda = iter(self.lambd[:,yi].tolist())
        iter_a = iter(self.a[:,yi].tolist())
        strings = []
        for i in range(len(self.dimensions)):
            temp ="F_i{i} = ".format(i=i+1)
            a = next(iter_a)
            for j in range(self.dimensions[i]):    
                for k in range(self.degrees[i]):
                    lambd = next(iter_lambda)
                    temp += '{0:+}'.format(lambd*a) + "T(x)^{k}".format(k=k)
            strings.append(temp)
        return strings
    def form_Fi(self):
        results={}
        for i in range(self.Y.shape[1]):
            results["For Y{i}".format(i=i+1)] = self.form_Fi_i(i)
        return results

    def form_Y_i(self,yi):
        iter_lambda = iter(self.lambd[:,yi].tolist())
        iter_a = iter(self.a[:,yi].tolist())
        iter_c = iter(self.c[:,yi].tolist())
        strings = [] 
        temp ="Y_i{i} = ".format(i=yi)
        for i in range(len(self.dimensions)):
            c = next(iter_c)
            for j in range(self.dimensions[i]): 
                a = next(iter_a)
                for k in range(self.degrees[i]):
                    lambd = next(iter_lambda)
                    temp += '{0:+}'.format(lambd*a*c) + "T(x)^{k}".format(k=k)
        strings.append(temp)
        return strings
    def form_Y(self):
        results={}
        for i in range(self.Y.shape[1]):
            results["For Y{i}".format(i=i+1)] = self.form_Y_i(i)
        return results
    def form_standart_Y_i(self,yi):
        iter_lambda = iter(self.lambd[:,yi].tolist())
        iter_a = iter(self.a[:,yi].tolist())
        iter_c = iter(self.c[:,yi].tolist())
        strings = [] 
        temp ="Y_i{i} = ".format(i=yi)
        for i in range(len(self.dimensions)):
            c = next(iter_c)
            for j in range(self.dimensions[i]): 
                a = next(iter_a)
                for k in range(self.degrees[i]):
                    lambd = next(iter_lambda)
                    temp += transform_to_standard(self.poly_type,k,lambd*a*c)
        strings.append(temp)
        return strings
    def form_standart_Y(self):
        results={}
        for i in range(self.Y.shape[1]):
            results["For Y{i}".format(i=i+1)] = self.form_standart_Y_i(i)
        return results
 

