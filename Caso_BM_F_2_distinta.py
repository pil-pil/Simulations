from __future__ import division
from scipy.integrate import quad
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fsolve
from scipy import optimize
from sympy import *
import multiprocessing

#Fondos objetivo
b_1 = 0.26
b_2 = 0.26

#Densidades de las preferencias
f_1 = lambda a_1 : 1
def f_2(x):
    if 0 <= x <= 1/2:
        fun = 80*x**4
    elif 1/2< x <= 1:
        fun = 80*(x-1)**4
    else:
        fun = 0
    return fun

#Distribuciones de las preferencias
F_1 = lambda a_1 : a_1
def F_2(a_2):
    distr = quad(f_2,0,a_2)[0]
    return distr

#Densidad sobre el ingreso futuro. La misma para ambos agentes emisores, a_i es la preferencia de los inversores al respecto del emisor i
g_1 = lambda x,a_1 : 3*(1-a_1)*(x-1)**2 + 3*a_1*x**2
g_2 = lambda x,a_2 : 3*(1-a_2)*(x-1)**2 + 3*a_2*x**2

#Esperanza del ingreso futuro dependiendo de la preferencia
es_1 = lambda x,a_1 : x*g_1(x,a_1)
es_2 = lambda x,a_2 : x*g_2(x,a_2)

def Esp_1(a_1):
    ans, err = quad(es_1,0,1, args = (a_1))
    return ans

def Esp_2(a_2):
    ans, err = quad(es_2,0,1, args = (a_2))
    return ans

a_1=np.linspace(0,1,50)
Esp_v_1 = np.vectorize(Esp_1)
#plt.plot(a_1, Esp_v_1(a_1))
#plt.show()

#Definicion del pago esperado por los inversores
def varphi_1(p_1,a_1):
    if isinstance(a_1, np.ndarray):
        a_1 = a_1[0]
    v_1 = lambda x,a_1,p_1 : (p_1/b_1)*g_1(x,a_1)*x
    ans_0 = quad(v_1,0,(b_1/p_1), args=(a_1,p_1))[0]
    ans_1 = quad(g_1,(b_1/p_1),1, args=(a_1))[0]
    return ans_0 + ans_1

def varphi_2(p_2,a_2):
    if isinstance(a_2, np.ndarray):
        a_2 = a_2[0]
    v_2 = lambda x,a_2,p_2 : (p_2/b_2)*g_2(x,a_2)*x
    ans_0, err_0 = quad(v_2,0,(b_2/p_2), args=(a_2,p_2))
    ans_1, err_1 = quad(g_2,(b_2/p_2),1, args=(a_2))
    return ans_0 + ans_1

#Definicion teta_barra
teta_1_barra = fsolve(lambda a_1: Esp_1(a_1) - b_1, 0.5)[0]
teta_2_barra = fsolve(lambda a_2: Esp_2(a_2) - b_2, 0.5)[0]

#Definicion de los teta Gorro
teta_1_gorro_aux = lambda p_1 : fsolve(lambda a_1 : (varphi_1(p_1,a_1) -p_1), 0.5)[0]
teta_2_gorro_aux = lambda p_2 : fsolve(lambda a_2 : (varphi_2(p_2,a_2) -p_2), 0.5)[0]
teta_1_gorro = lambda p_1 : min(1,(max(0,teta_1_gorro_aux(p_1))))
teta_2_gorro = lambda p_2 : min(1,(max(0,teta_2_gorro_aux(p_2))))

#Grafica del pago esperado por los bonos de 1 para un inversor tipo 0 y tipo 1
p_1 = np.linspace(b_1,1,50)
varphi_1_plot_0 = lambda p_1 : varphi_1(p_1,0)
varphi_1_plot_1 = lambda p_1 : varphi_1(p_1,1)
varphi_1_plot_v_0 = np.vectorize(varphi_1_plot_0)
varphi_1_plot_v_1 = np.vectorize(varphi_1_plot_1)
#plt.plot(p_1,varphi_1_plot_v_0(p_1))
#plt.plot(p_1,varphi_1_plot_v_1(p_1))
#plt.show()

#Definicion de alfa y su inversa
def alfa(a_1,p_1,p_2):
    alf = fsolve(lambda a_2 : (varphi_1(p_1,a_1)/p_1) - (varphi_2(p_2,a_2)/p_2), 0.5)[0]
    alf = min(alf, 1)
    return alf

def inv_alfa(a_2,p_1,p_2):
    inv_alf = fsolve(lambda a_1 : (varphi_1(p_1,a_1)/p_1) - (varphi_2(p_2,a_2)/p_2), 0.5)[0]
    inv_alf = min(max(inv_alf,0),1)
    return inv_alf

#Grafica de alfa en funcion de teta_1 con los demas parametros fijos
a_1 = np.linspace(0,1,50)
alfa_plot = lambda a_1 : alfa(a_1,0.3,0.5)
alfa_plot_v = np.vectorize(alfa_plot)
#plt.plot(a_1, alfa_plot_v(a_1))
#plt.show()

#Funciones auxiliares para definir las demandas
aux_1 = lambda p_1 : max(teta_1_gorro(p_1),teta_1_barra)
aux_2 = lambda p_2 : max(teta_2_gorro(p_2),teta_2_barra)

def B(p_1,p_2):
    if inv_alfa(1,p_1,p_2) < teta_1_barra:
        B = 0
    else:
        B = 1
    return B
#Funciones de demanda
def d_2(p_1,p_2):
    integrando = lambda a_1,a_2: f_1(a_1)*f_2(a_2)
    lim_inf_1 = aux_1(p_1)
    lim_sup_1 = min(inv_alfa(1,p_1,p_2),1)
    lim_inf_2 = lambda a_1 :  max(teta_2_barra,alfa(a_1,p_1,p_2))
    lim_sup_2 = lambda a_1 : 1
    dem_2_int = dblquad(integrando, lim_inf_1, lim_sup_1, lim_inf_2, lim_sup_2)[0]
    dem_2 = (1- F_2(aux_2(p_2)))*(F_1(aux_1(p_1))) + B(p_1,p_2)*dem_2_int
    return dem_2

d_0 = lambda p_1,p_2 : F_1(aux_1(p_1))*F_2(aux_2(p_2))
d_1 = lambda p_1,p_2: 1 - d_0(p_1,p_2) - d_2(p_1,p_2)

#Funciones de mejor respuestas

def mej_resp_1(p_2):
    p_1_m = lambda p_1 : d_1(p_1,p_2) -b_1
    p_1_mej = fsolve(p_1_m, 0.7)[0]
    return p_1_mej

def mej_resp_2(p_1):
    p_2_m = lambda p_2 : d_2(p_1,p_2) -b_2
    p_2_mej = fsolve(p_2_m, 0.7)[0]
    return p_2_mej

def H(x):
    H_1 = mej_resp_1(x[1]) - x[0]
    H_2 = mej_resp_2(x[0]) - x[1]
    return [H_1,H_2]

#Encontrar el equilibrio
x_1_sample = np.linspace(0,1,5)
x_2_sample = np.linspace(0,1,5)

lista_equilibrios = []

def L(x_1,x_2, return_dict):
    x = fsolve(H,[x_1,x_2])
    p_1_estrella = x[0]
    p_2_estrella = x[1]
    return_dict[f"{x_1},{x_2}"]=(p_1_estrella,p_2_estrella)


if __name__ == '__main__':

    #Grafica de alfa en funcion de teta_1 con los demas parametros fijos
    p_1 = np.linspace(b_1,1,50)
    d_1_plot = lambda p_1 : d_1(p_1,0.3)
    #d_1_plot_v = np.vectorize(d_1_plot)
    #plt.show()

    p_2 = np.linspace(b_2,1,50)
    d_2_plot = lambda p_2 : d_2(0.3,p_2)
    d_2_plot_v = np.vectorize(d_2_plot)
    #plt.plot(p_2, d_2_plot_v(p_2))
    #plt.show()

    #Graficas mejor respuesta
    p_2 = np.linspace(b_2,1,50)
    mej_resp_1_plot = lambda p_2 : min(mej_resp_1(p_2),1)
    mej_resp_1_plot_v = np.vectorize(mej_resp_1_plot)
    #plt.plot(p_2, mej_resp_1_plot_v(p_2))
    #plt.show()

    p_1 = np.linspace(b_1,1,50)
    mej_resp_2_plot = lambda p_1 : min(mej_resp_2(p_1),1)
    mej_resp_2_plot_v = np.vectorize(mej_resp_2_plot)
    #plt.plot(p_1, mej_resp_1_plot_v(p_1))
    #plt.show()

    #Calculo del equilibrio
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for x_1 in x_1_sample:
        for x_2 in x_2_sample:
            p = multiprocessing.Process(target=L, args=(x_1,x_2,return_dict))
            processes.append(p)
            p.start()

    for process in processes:
        process.join()

    for i in return_dict:
        x = return_dict[i]
        p_1 = x[0]
        p_2 = x[1]
        print(return_dict[i])
        if (0< p_1 <=1) and (0< p_2 <=1):
            lista_equilibrios.append(return_dict[i])

    print(lista_equilibrios)
