#Modified Mathieu Funtions of first (Ie and Io) and second (Ke and Ko) kind
#28.24.6-28.24.13 DLMF

import scipy.special as sp
import numpy as np
gamma = 0.5772156649015328606065120900824024310421


def Ie(m,q,z):
    h=np.sqrt(q)
    if m%2==0:
        if m==0:
            eps_s=2
        else:
            eps_s=1
        suma=0
        for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
            suma += (-1.)**i*sp.mathieu_even_coef(m,q)[i]*(sp.iv(i-m//2,h*np.exp(-z))\
                    *sp.iv(i+m//2,h*np.exp(z))+sp.iv\
                    (i+m//2,h*np.exp(-z))*sp.iv(i-m//2,h*np.exp(z)))
        Ie = (-1.)**(m//2)*1/sp.mathieu_even_coef(m,q)[m//2]*suma*1/eps_s
    else:
        suma = 0
        for i in np.arange(0,len(sp.mathieu_odd_coef(m,q)),1):
            suma += (-1)**i*sp.mathieu_odd_coef(m,q)[i]*(sp.iv(i-(m-1)//2,h*np.exp(-z))\
                    *sp.iv(i+(m-1)//2+1,h*np.exp(z))+sp.iv(i+(m-1)//2+1,h*np.exp(-z))\
                    *sp.iv(i-(m-1)//2,h*np.exp(z)))
        Ie = (-1)**((m-1)//2)*1/sp.mathieu_odd_coef(m,q)[(m-1)//2]*suma
    return Ie

def Io(m,q,z):
    h=np.sqrt(q)
    if m%2==0:
        suma=0
        for i in np.arange(0,len(sp.mathieu_odd_coef(m,q)),1):
            suma += (-1)**i*sp.mathieu_odd_coef(m,q)[i]*(sp.iv(i-(m-2)//2,h*np.exp(-z))\
                    *sp.iv(i+(m-2)//2+2,h*np.exp(z))-sp.iv\
                    (i+(m-2)//2+2,h*np.exp(-z))*sp.iv(i-(m-2)//2,h*np.exp(z)))
        Io = (-1)**m*1/sp.mathieu_odd_coef(m,q)[m//2]*suma
    else:
        suma = 0
        for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
            suma += (-1)**i*sp.mathieu_even_coef(m,q)[i]*(sp.iv(i-(m-1)//2,h*np.exp(-z))\
                    *sp.iv(i+(m-1)//2+1,h*np.exp(z))-sp.iv(i+(m-1)//2+1,h*np.exp(-z))\
                    *sp.iv(i-(m-1)//2,h*np.exp(z)))
        Io = (-1)**m*1/sp.mathieu_even_coef(m,q)[(m-1)//2]*suma
    return Io

def Ke(m,q,z):
    h=np.sqrt(q)
    if m%2==0:
        if m==0:
            eps_s=2
        else:
            eps_s=1
        suma=0
        for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
            suma += sp.mathieu_even_coef(m,q)[i]*(sp.iv(i-m//2,h*np.exp(-z))\
                    *sp.kv(i+m//2,h*np.exp(z))+sp.iv\
                    (i+m//2,h*np.exp(-z))*sp.kv(i-m//2,h*np.exp(z)))
        Ke = 1/sp.mathieu_even_coef(m,q)[m//2]*suma*1/eps_s
    else:
        suma = 0
        for i in np.arange(0,len(sp.mathieu_odd_coef(m,q)),1):
            suma += sp.mathieu_odd_coef(m,q)[i]*(sp.iv(i-(m-1)//2,h*np.exp(-z))\
                    *sp.kv(i+(m-1)//2+1,h*np.exp(z))-sp.iv(i+(m-1)//2+1,h*np.exp(-z))\
                    *sp.kv(i-(m-1)//2,h*np.exp(z)))
        Ke = 1/sp.mathieu_odd_coef(m,q)[(m-1)//2]*suma
    return Ke

def Ko(m,q,z):
    h=np.sqrt(q)
    if m%2==0:
        suma=0
        for i in np.arange(0,len(sp.mathieu_odd_coef(m,q)),1):
            suma += sp.mathieu_odd_coef(m,q)[i]*(sp.iv(i-(m-2)//2,h*np.exp(-z))\
                    *sp.kv(i+(m-2)//2+2,h*np.exp(z))-sp.iv\
                    (i+(m-2)//2+2,h*np.exp(-z))*sp.kv(i-(m-2)//2,h*np.exp(z)))
        Ko = 1/sp.mathieu_odd_coef(m,q)[m//2]*suma
    else:
        suma = 0
        for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
            suma += sp.mathieu_even_coef(m,q)[i]*(sp.iv(i-(m-1)//2,h*np.exp(-z))\
                    *sp.kv(i+(m-1)//2+1,h*np.exp(z))+sp.iv(i+(m-1)//2+1,h*np.exp(-z))\
                    *sp.kv(i-(m-1)//2,h*np.exp(z)))
        Ko = 1/sp.mathieu_even_coef(m,q)[(m-1)//2]*suma
    return Ko

def Kep(m,q,z):
    h=np.sqrt(q)
    if m%2==0:
        if m==0:
            eps_s=2
        else:
            eps_s=1
        suma=0
        for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
            suma += sp.mathieu_even_coef(m,q)[i]*(bessel_product_2m_1(i,m//2,h,z)\
                    +bessel_product_2m_2(i,m//2,h,z))
        Ke = 1/sp.mathieu_even_coef(m,q)[m//2]*suma*1/eps_s
    else:
        suma = 0
        for i in np.arange(0,len(sp.mathieu_odd_coef(m,q)),1):
            suma += sp.mathieu_odd_coef(m,q)[i]*(bessel_product_2m1_1(i,(m-1)//2,h,z)\
                    -bessel_product_2m1_2(i,(m-1)//2,h,z))
        Ke = 1/sp.mathieu_odd_coef(m,q)[(m-1)//2]*suma
    return Ke

def Kop(m,q,z):
    h=np.sqrt(q)
    if m%2==0:
        suma=0
        for i in np.arange(0,len(sp.mathieu_odd_coef(m,q)),1):
            suma += sp.mathieu_odd_coef(m,q)[i]*(bessel_product_2m2_1(i,(m-2)//2,h,z)\
            -bessel_product_2m2_2(i,(m-2)//2,h,z))
        Kop = 1/sp.mathieu_odd_coef(m,q)[m//2]*suma
    else:
        suma = 0
        for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
            suma += sp.mathieu_even_coef(m,q)[i]*(bessel_product_2m1_1(i,(m-1)//2,h,z)\
                    +bessel_product_2m1_2(i,(m-1)//2,h,z))
        Kop = 1/sp.mathieu_even_coef(m,q)[(m-1)//2]*suma
    return Kop

def bessel_product_2m_1(i,s,h,z):
    return -h*np.exp(-z)*(sp.iv(i-s+1,h*np.exp(-z))+\
            (i-s)*sp.iv(i-s,h*np.exp(-z))/(h*np.exp(-z)))\
            *sp.kv(i+s,h*np.exp(z)) + sp.iv(i-s\
            ,h*np.exp(-z))*h*np.exp(z)*(-sp.kv(i+s+1,h*np.exp(z))\
            +(i+s)*sp.kv(i+s,h*np.exp(z))/(h*np.exp(z)))

def bessel_product_2m_2(i,s,h,z):
    return -h*np.exp(-z)*(sp.iv(i+s+1,h*np.exp(-z))+\
            (i+s)*sp.iv(i+s,h*np.exp(-z))/(h*np.exp(-z)))\
            *sp.kv(i-s,h*np.exp(z)) + sp.iv(i+s\
            ,h*np.exp(-z))*h*np.exp(z)*(-sp.kv(i-s+1,h*np.exp(z))\
            +(i-s)*sp.kv(i-s,h*np.exp(z))/(h*np.exp(z)))

def bessel_product_2m1_1(i,s,h,z):
    return h*np.exp(z)*sp.iv(i-s,h*np.exp(-z))*(\
            -sp.kv(i+s,h*np.exp(z))-(i+s+1)*sp.kv(\
            i+s+1,h*np.exp(z))/(h*np.exp(z)))-sp.kv(i+s+1,h*np.exp(z))\
            *h*np.exp(-z)*(sp.iv(i-s+1,h*np.exp(-z))+(i-s)\
            *sp.iv(i-s,h*np.exp(-z))/(h*np.exp(-z)))
            
def bessel_product_2m1_2(i,s,h,z):
    return h*np.exp(z)*sp.iv(i+s+1,h*np.exp(-z))*(\
            -sp.kv(i-s+1,h*np.exp(z))+(i-s)*sp.kv(\
            i-s,h*np.exp(z))/(h*np.exp(z)))-sp.kv(i-s,h*np.exp(z))\
            *h*np.exp(-z)*(sp.iv(i+s,h*np.exp(-z))-(i+s+1)\
            *sp.iv(i+s+1,h*np.exp(-z))/(h*np.exp(-z)))

def bessel_product_2m2_1(i,s,h,z):
    return -h*np.exp(-z)*sp.kv(i+s+2,h*np.exp(z))*(sp.iv(\
            i-s+1,h*np.exp(-z))+(i-s)*sp.iv(i-s,h*np.exp(-z))/(h*np.exp(-z)))\
            +sp.iv(i-s,h*np.exp(-z))*h*np.exp(z)*(-sp.kv(\
            i+s+1,h*np.exp(z))-(i+s+2)*sp.kv(i+s+2,h*np.exp(z))/\
            (h*np.exp(z)))

def bessel_product_2m2_2(i,s,h,z):
    return -h*np.exp(-z)*sp.kv(i-s,h*np.exp(z))*(sp.iv(\
            i+s+1,h*np.exp(-z))-(i+s+2)*sp.iv(i+s+2,h*np.exp(-z))\
            /(h*np.exp(-z)))+sp.iv(i+s+2,h*np.exp(-z))*h*np.exp(-z)*(\
             -sp.kv(i-s+1,h*np.exp(z))+(i-s)*sp.kv(\
            i-s,h*np.exp(z))/(h*np.exp(z)))

def bessel_product_ie0(i,s,h,z):
    return -h*np.exp(-z)*sp.iv(i,h*np.exp(z))*(sp.iv(i+1,h*np.exp(-z))\
        +i/h*np.exp(z)*sp.iv(i,h*np.exp(-z)))+h*np.exp(z)*sp.iv(i,h*np.exp(-z))\
        *(sp.iv(i+1,h*np.exp(z))+i/h*np.exp(-z)*sp.iv(i,h*np.exp(z)))

def iep0(m,q,z):
    h = np.sqrt(q)
    suma = 0
    for i in np.arange(0,len(sp.mathieu_even_coef(m,q)),1):
        suma += (-1)**i*sp.mathieu_even_coef(m,q)[i]/sp.mathieu_even_coef(m,q)[0]\
        *bessel_product_ie0(i,m,h,z)
    return suma

def steady_solution_sepvbles(rho,theta,n_sepvbles,a,b,Pe,dirichlet):

    # # # # # # # # # # # IMPORTANT n_sepvbles MUST BE EVEN

    # B and D coefficients (imposed by boundary conditions)


    # coefficients of the solution
    B = np.zeros(n_sepvbles)
    D = np.zeros(n_sepvbles)

    # intermediate coefficients to define B and D
    alpha = np.zeros(n_sepvbles)
    beta = np.zeros(n_sepvbles)

    for n in np.arange(0,n_sepvbles//2,1):
        alpha_even=0
        alpha_odd=0
        # # # # number of fourier coeffs in ce and se functions
        n_ce = len(special.mathieu_even_coef(2*n,Pe**2/4))
        for l in np.arange(0,n_sepvbles,1):
            for p in np.arange(0,n_ce,1):
                alpha_even += (-1)**p*a[l]*special.mathieu_even_coef(2*n,Pe**2/4)[p]\
                *(special.iv(l+2*p,-Pe)+special.iv(l-2*p,-Pe))
                alpha_odd += (-1)**p*a[l]*special.mathieu_odd_coef(2*n+1,Pe**2/4)[p]\
                *(special.iv(l+2*p+1,-Pe)+special.iv(l-2*p-1,-Pe))

        alpha[2*n] = (-1)**n*alpha_even
        alpha[2*n+1]=(-1)**n*alpha_odd

    for n in np.arange(0,n_sepvbles//2,1):
        beta_odd=0
        beta_even=0
        n_ce = len(special.mathieu_even_coef(2*n+1,Pe**2/4))
        for l in np.arange(0,n_sepvbles,1):
            for p in np.arange(0,n_ce,1):
                beta_odd += (-1)**p*b[l]*special.mathieu_even_coef(2*n+1,Pe**2/4)[p]\
                *(-special.iv(l+2*p+1,-Pe)+special.iv(l-2*p-1,-Pe))
                beta_even += (-1)**p*b[l]*special.mathieu_odd_coef(2*n+2,Pe**2/4)[p]\
                *(-special.iv(l+2*p+2,-Pe)+special.iv(l-2*p-2,-Pe))

        beta[2*n+1]=(-1)**n*beta_odd
        if n < n_sepvbles//2-1:
            beta[2*n+2] = (-1)**n*beta_even

    B = np.zeros([n_sepvbles,1],dtype=complex)
    D = np.zeros([n_sepvbles,1],dtype=complex)

    if dirichlet==1:
        for i in np.arange(0,n_sepvbles,1):
            B[i] = alpha[i] * 1/Ke(i,Pe**2/4,0)
        for i in np.arange(1,n_sepvbles,1):
            D[i] = beta[i] * 1/Ko(i,Pe**2/4,0)
    else:
        for i in np.arange(0,n_sepvbles,1):
            B[i] = alpha[i] * 1/Kep(i,Pe**2/4,0)
        for i in np.arange(1,n_sepvbles,1):
            D[i] = beta[i] * 1/Kop(i,Pe**2/4,0)

    mathieu_sol = np.zeros(rho.shape,dtype=complex)

    mathieu_sol = np.exp(Pe/2*np.cos(theta)*(rho+1/rho))*B[0]*Ke(0,Pe**2/4,np.log(rho))\
        *special.mathieu_cem(0,-Pe**2/4,theta*180/np.pi)[0]

    for i in np.arange(1,n_sepvbles,1,dtype=int):
        mathieu_sol += np.exp(Pe/2*np.cos(theta)*(rho+1/rho))*B[i]*Ke(i,Pe**2/4,np.log(rho))\
        *special.mathieu_cem(i,-Pe**2/4,theta*180/np.pi)[0]\
        +np.exp(Pe/2*np.cos(theta)*(rho+1/rho))*D[i]*Ko(i,Pe**2/4,np.log(rho))\
        *special.mathieu_sem(i,-Pe**2/4,theta*180/np.pi)[0]

    return mathieu_sol