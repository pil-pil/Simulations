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
b_1 = 0.45
b_2 = 0.45

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
f_aux = lambda x : 1 - (x/b_2)
K = quad(f_aux,0,b_2)[0]
teta_3_barra = fsolve(lambda a_2 : varphi_2(1,a_2) - 1 + K, 0.5)[0]

#Definicion de los teta Gorro
teta_1_gorro_aux = lambda p_1 : fsolve(lambda a_1 : (varphi_1(p_1,a_1) -p_1), 0.5)[0]
teta_2_gorro_aux = lambda p_2 : fsolve(lambda a_2 : (varphi_2(p_2,a_2) -p_2), 0.5)[0]
teta_3_gorro_aux = lambda p_2,r: fsolve(lambda a_2 : 1 - varphi_2(p_2,a_2) - r, 0.5)[0]

teta_1_gorro = lambda p_1 : min(1,(max(0,teta_1_gorro_aux(p_1))))
teta_2_gorro = lambda p_2 : min(1,(max(0,teta_2_gorro_aux(p_2))))
teta_3_gorro = lambda p_2,r : min(1,(max(0,teta_3_gorro_aux(p_2,r))))


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

def beta(a_1,p_1,p_2,r):
    bet = fsolve(lambda a_2 : ((1-varphi_2(p_2,a_2))/r) - (varphi_1(p_1,a_1)/p_1), 0.5)[0]
    bet = max(bet,0)
    return bet

def inv_beta(a_2,p_1,p_2,r):
    inv_bet = fsolve(lambda a_1 : ((1-varphi_2(p_2,a_2))/r) - (varphi_1(p_1,a_1)/p_1), 0.5)[0]
    inv_bet = min(max(inv_bet,0),1)
    return inv_bet

#Definicion de los teta_tilde
def teta_2_tilde(p_2,r):
     tet_tild = fsolve(lambda a_2 : varphi_2(p_2,a_2)/p_2 - (1-varphi_2(p_2,a_2))/r, 0.5)[0]
     tet_tild = min(max(tet_tild,0),1)
     return tet_tild
teta_1_tilde_aux = lambda p_1,p_2,r : inv_alfa(teta_2_tilde(p_2,r),p_1,p_2)
teta_1_tilde = lambda p_1,p_2,r: min(teta_1_tilde_aux(p_1,p_2,r),1)

#Grafica de alfa en funcion de teta_1 con los demas parametros fijos
a_1 = np.linspace(0,1,50)
alfa_plot = lambda a_1 : alfa(a_1,0.3,0.5)
alfa_plot_v = np.vectorize(alfa_plot)
#plt.plot(a_1, alfa_plot_v(a_1))
#plt.show()

a_1 = np.linspace(0,1,50)
beta_plot = lambda a_1 : beta(a_1,0.3,0.5,0.3)
beta_plot_v = np.vectorize(beta_plot)
#plt.plot(a_1, beta_plot_v(a_1))
#plt.show()

#print(f"Teta_2_tilde(0.5,0.3) : {teta_2_tilde(0.5,0.3)}")
#print(f"Teta_1_tilde(0.3,0.5,0.3) : {teta_1_tilde(0.3,0.5,0.3)}")


#Funciones auxiliares para definir las demandas
f_1 = lambda p_1 : max(teta_1_gorro(p_1),teta_1_barra)
f_2 = lambda p_2 : max(teta_2_gorro(p_2),teta_2_barra)
f_3 = lambda p_1,p_2 : (1-f_1(p_1))*(1-teta_2_barra)
f_1_tilde_aux = lambda p_1,p_2,r: max(teta_1_barra, inv_alfa(teta_2_barra,p_1,p_2))
f_1_tilde = lambda p_1,p_2,r: max(f_1_tilde_aux(p_1,p_2,r), teta_1_tilde(p_1,p_2,r))
f_2_tilde = lambda p_2,r: max(teta_2_barra,teta_2_tilde(p_2,r))
f_3_tilde = lambda p_2,r: min(teta_3_barra,teta_2_tilde(p_2,r))
def f_4(p_1,p_2):
    integrando = lambda a_1,a_2 : 1
    lim_inf_1 = f_1(p_1)
    lim_sup_1 = min(inv_alfa(1,p_1,p_2),1)
    lim_inf_2 = lambda a_1 : max(teta_2_barra,alfa(a_1,p_1,p_2))
    lim_sup_2 = lambda a_1 : 1
    int_f_4 = dblquad(integrando, lim_inf_1, lim_sup_1, lim_inf_2, lim_sup_2)[0]
    return int_f_4
f_5 = lambda p_2,r : min(teta_3_gorro(p_2,r), teta_3_barra)
def f_6(p_1,p_2,r):
    integrando = lambda a_1,a_2 : 1
    lim_inf_1 = f_1(p_1)
    lim_sup_1 = min(inv_beta(0,p_1,p_2,r),1)
    lim_inf_2 = lambda a_1 : 0
    lim_sup_2 = lambda a_1 : min(beta(a_1,p_1,p_2,r),teta_3_barra)
    int_f_6 = dblquad(integrando, lim_inf_1, lim_sup_1, lim_inf_2, lim_sup_2)[0]
    return int_f_6
def f_7(p_1,p_2,r):
    integrando = lambda a_1,a_2 : 1
    lim_inf_1 = f_1(p_1)
    lim_sup_1 = min(inv_alfa(1,p_1,p_2),1)
    lim_inf_2 = lambda a_1 : max(f_2_tilde(p_2,r),alfa(a_1,p_1,p_2))
    lim_sup_2 = lambda a_1 : 1
    int_f_7 = dblquad(integrando, lim_inf_1, lim_sup_1, lim_inf_2, lim_sup_2)[0]
    return int_f_7
def f_8(p_1,p_2,r):
    integrando = lambda a_1,a_2 : 1
    lim_inf_1 = f_1(p_1)
    lim_sup_1 = min(inv_beta(0,p_1,p_2,r),1)
    lim_inf_2 = lambda a_1 : 0
    lim_sup_2 = lambda a_1 : min(beta(a_1,p_1,p_2,r), f_3_tilde(p_2,r))
    int_f_8 = dblquad(integrando, lim_inf_1, lim_sup_1, lim_inf_2, lim_sup_2)[0]
    return int_f_8


#Demandas cuando teta_3_gorro < teta_2_gorro
def H_2(p_1,p_2):
    if alfa(1,p_1,p_2) <= teta_2_barra:
        h_2 = f_3(p_1,p_2)
    elif inv_alfa(1,p_1,p_2) <= teta_1_barra:
        h_2 = 0
    else:
        h_2 = f_4(p_1,p_2)
    return h_2

def H_3(p_1,p_2,r):
    if beta(1,p_1,p_2,r) >= teta_3_barra:
        h_3 = (1-f_1(p_1))*f_5(p_2,r)
    elif inv_beta(0,p_1,p_2,r) <= teta_1_barra:
        h_3 = 0
    else:
        h_3 = f_6(p_1,p_2,r)
    return h_3



d_2 = lambda p_1,p_2,r: f_1(p_1)*(1-f_2(p_2)) + H_2(p_1,p_2)
d_3 = lambda p_1,p_2,r: f_1(p_1)*f_5(p_2,r) + H_3(p_1,p_2,r)
d_1 = lambda p_1,p_2,r: 1 - d_2(p_1,p_2,r) - d_3(p_1,p_2,r) - ((f_2(p_2) - min(teta_3_gorro(p_2,r), teta_3_barra))*(f_1(p_1)))

#Demandas con teta_2_gorro < teta_3_gorro
def J_2(p_1,p_2,r):
    if alfa(1,p_1,p_2) <= f_2_tilde(p_2,r):
        j_2 = (1-f_2_tilde(p_2,r))*(1-f_1(p_1))
    elif inv_alfa(1,p_1,p_2) <= teta_1_barra:
        j_2 = 0
    else:
        j_2 = f_7(p_1,p_2,r)
    return j_2

def J_3(p_1,p_2,r):
    if beta(1,p_1,p_2,r) >= f_3_tilde(p_2,r):
        j_3 = (f_3_tilde(p_2,r))*(1-f_1(p_1))
    elif inv_beta(1,p_1,p_2,r) <= teta_1_barra:
        j_3 = 0
    else:
        j_3 = f_8(p_1,p_2,r)
    return j_3

D_2 = lambda p_1,p_2,r: (f_1(p_1))*(1-f_2_tilde(p_2,r)) + J_2(p_1,p_2,r)
D_3 = lambda p_1,p_2,r: (f_1(p_1))*(f_3_tilde(p_2,r)) + J_3(p_1,p_2,r)
D_1 = lambda p_1,p_2,r : 1- D_2(p_1,p_2,r) - D_3(p_1,p_2,r) - ((f_1(p_1))*(min(0, teta_2_barra - teta_2_tilde(p_2,r)) + min(teta_2_tilde(p_2,r)-teta_3_barra,0)))

#Demandas en caso general
def Demanda_2(p_1,p_2,r):
    if teta_3_gorro(p_2,r) <= teta_2_gorro(p_2):
        Dem_2 = d_2(p_1,p_2,r)
    else:
        Dem_2 = D_2(p_1,p_2,r)
    return Dem_2

def Demanda_3(p_1,p_2,r):
    if teta_3_gorro(p_2,r) <= teta_2_gorro(p_2):
        Dem_3 = d_3(p_1,p_2,r)
    else:
        Dem_3 = D_3(p_1,p_2,r)
    return Dem_3

def Demanda_1(p_1,p_2,r):
    if teta_3_gorro(p_2,r) <= teta_2_gorro(p_2):
        Dem_1 = d_1(p_1,p_2,r)
    else:
        Dem_1 = D_1(p_1,p_2,r)
    return Dem_1

#Graficas caso 1
p_2 = np.linspace(b_2,1,50)
dem_2_plot = lambda p_2 : d_2(0.3,p_2,0.7)
dem_2_plot_v = np.vectorize(dem_2_plot)
#plt.plot(p_2,dem_2_plot_v(p_2))
#plt.show()

p_1 = np.linspace(b_1,1,50)
dem_1_plot = lambda p_1 : d_1(p_1,0.5,0.7)
dem_1_plot_v = np.vectorize(dem_1_plot)
#plt.plot(p_1,dem_1_plot_v(p_1))
#plt.show()

def K(p_2):
    integrando = lambda x : 1 - min(1, x*p_2/b_2)
    int = quad(integrando, 0,1)[0]
    return int

r = np.linspace(K(0.3),1,50)
dem_3_plot = lambda r : d_3(0.3,0.3,r)
dem_3_plot_v = np.vectorize(dem_3_plot)
#plt.plot(r,dem_3_plot_v(r))
#plt.show()

#Graficas caso general
p_2 = np.linspace(b_2,1,50)
Dem_2_plot = lambda p_2 : Demanda_2(0.3,p_2,0.7)
Dem_2_plot_v = np.vectorize(Dem_2_plot)
#plt.plot(p_2,Dem_2_plot_v(p_2))
#plt.show()

r = np.linspace(K(0.3),1,50)
Dem_3_plot = lambda r : D_3(0.3,0.3,r)
Dem_3_plot_v = np.vectorize(Dem_3_plot)
#plt.plot(r,Dem_3_plot_v(r))
#plt.show()

p_1 = np.linspace(b_1,1,50)
Dem_1_plot = lambda p_1 : Demanda_1(p_1,0.3,0.7)
Dem_1_plot_v = np.vectorize(Dem_1_plot)
#plt.plot(p_1,Dem_1_plot_v(p_1))
#plt.show()

#Utilidades para los agentes
u_1 = lambda p_1: max(0, 1/2 + (b_1/p_1)**2/2 -(b_1/p_1))
u_2 = lambda p_2: max(0, 1/2 + (b_2/p_2)**2/2 -(b_2/p_2))
u_3 = lambda r,s,p_2 : s*(r - K(p_2))

def R(p_1,p_2,r):
    R_op = u_3(r, Demanda_3(p_1,p_2,r)/r, p_2)
    return R_op

#Mejores respuestas
def armax(p_1,p_2):
    R_min = lambda r : - R(p_1,p_2,r)
    armx = optimize.fmin(R_min,0.5)
    armx = armx[0]
    return armx

def p_1_mej_resp(p_2,r):
    deman_1 = lambda p_1 : Demanda_1(p_1,p_2,r)-b_1
    a = fsolve(deman_1,0.6)[0]
    return a

def p_2_mej_resp(p_1,r):
    deman_2 = lambda p_2 : Demanda_2(p_1,p_2,r)-b_2
    a = fsolve(deman_2,0.6)[0]
    return a

def H(x):
    H_1 = p_1_mej_resp(x[1],x[2]) - x[0]
    H_2 = p_2_mej_resp(x[0],x[2]) - x[1]
    H_3 = armax(x[0],x[1]) - x[2]
    return [H_1,H_2,H_3]

#Encontrar el equilibrio
x_1_sample = np.linspace(0,1,3)
x_2_sample = np.linspace(0,1,3)
x_3_sample = np.linspace(0,1,3)

lista_equilibrios = []

def L(x_1,x_2,x_3, return_dict):
    x = fsolve(H,[x_1,x_2,x_3])
    p_1_estrella = x[0]
    p_2_estrella = x[1]
    r_estrella = x[2]
    return_dict[f"{x_1},{x_2},{x_3}"]=(p_1_estrella,p_2_estrella,r_estrella)



if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for x_1 in x_1_sample:
        for x_2 in x_2_sample:
            for x_3 in x_3_sample:
                p = multiprocessing.Process(target=L, args=(x_1,x_2,x_3,return_dict))
                processes.append(p)
                p.start()

    for process in processes:
        process.join()

    for i in return_dict:
        x = return_dict[i]
        p_1 = x[0]
        p_2 = x[1]
        r = x[2]
        #print(return_dict[i])
        if b_1<= p_1 <=1 and b_2<= p_2 <=1 and K(p_2)<= r <=1:
            lista_equilibrios.append(return_dict[i])


    print(lista_equilibrios)
    print("ES ESTE con 0.6 en mej rpta y mejores funciones de demanda CORREGIDAS y b_1 = b_2 = 0.45")



#print(H(x))
#print(f"El equilibrio se da en ({p_1_estrella},{p_2_estrella},{r_estrella})")
#print(f"Las ganancias son: {u_3(r_estrella, Demanda_3(p_1_estrella,p_2_estrella,r_estrella)/r_estrella,p_2_estrella)}, {u_1(p_1_estrella)}, {u_2(p_2_estrella)}")
#print(f"Demanda de A_1 de equilibrio: {Demanda_1(p_1_estrella,p_2_estrella,r_estrella)}")
