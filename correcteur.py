from math import *
import numpy as np
import numpy.random as random
from pylab import *

m=0.468
g=9.81
I=[[4.856*(10**(-3)),0,0],[0,4.856*(10**(-3)),0],[0,0,4.801*(10**(-3))]]
k=2.980*(10**(-6))
kd=0.25
L=0.225
b=1.140*(10**(-7))
dt=0.005






def SommeElements(L):
    S=0
    for i in range(len(L)):
       S+=L[i]
    return S

def SommeListe(L):
    S=[]
    l=len(L)
    for i in range(len(L[0])):
        h=0
        for j in range(l):
            h+=L[j][i]
        S.append(h)
    return S
    
def SommeVecteur(L1,L2):
    M=[]
    for i in range(len(L1)):
        M.append(L1[i]+L2[i])
    return(M)
    
def mulmatscal(a,L):
    S=[]
    for i in range(len(L)):
        S.append(a*L[i])
    return S
        
    
def deg2rad(a):
    b=radians(a)
    return b
    
def rotation(angles):
    
    psi=deg2rad(angles[0][0])
    theta=deg2rad(angles[1][0])
    phi=deg2rad(angles[2][0])
    
    R=[[cos(phi)*cos(theta),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],[sin(phi)*cos(theta),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi)],[-sin(theta),cos(theta)*sin(psi),cos(theta)*cos(psi)]]
    
    return R
    
    
    
    
def thetadottoomega(angles):
  
    psi=deg2rad(angles[0])
    theta=deg2rad(angles[1])
    phi=deg2rad(angles[2])
  
    R=[[1,0,-sin(theta)],[0,cos(psi),cos(theta)*sin(psi)],[0,-sin(psi),cos(theta)*cos(psi)]]
    
    return R
    
    
    
    
def omegatothetadot(angles):
  
    psi=deg2rad(angles[0][0])
    theta=deg2rad(angles[1][0])
    phi=deg2rad(angles[2][0])
  
    R=[[1,sin(psi)*tan(theta),cos(psi)*tan(theta)],[0,cos(psi),-sin(psi)],[0,sin(psi)/cos(theta),cos(psi)/cos(theta)]]
    
    return R




def Poussee(inputs,k):
    P=0
    P=k*(SommeElements(inputs))
    T=[0,0,P]
    return T
    
def Couple(inputs, L, b, k):
    
    tau1=L*k*(inputs[0]-inputs[2])
    tau2=L*k*(inputs[1]-inputs[3])
    tau3=b*(inputs[0]-inputs[1]+inputs[2]-inputs[3])
    tau=[tau1,tau2,tau3]
    return tau
    
def acceleration(inputs,angles,xdot,m,g,k,kd):
    gravite=[0,0,-g]
    R=rotation(angles)
    P=Poussee(inputs,k)
    Pmat=[[P[0]],[P[1]],[P[2]]]
    Tmat=dot(R,Pmat)
    T=[Tmat[0][0],Tmat[1][0],Tmat[2][0]]
    Fd=mulmatscal(-kd,xdot)
    a=SommeListe([gravite, mulmatscal(1/m,T),Fd])
    return a
    
def angular_acceleration(inputs,omega,I,L,b,k):
    tau=Couple(inputs,L,b,k)
    # Iw=dot(I,omega)
    # Iwvect=[Iw[0][0],Iw[1][0],Iw[2][0]]
    # prodvect=np.cross(tau,Iwvect)
    # negatif_prodvect=mulmatscal(-1,prodvect)
    # piron=SommeListe([tau,negatif_prodvect])
    # mat_piron=[[piron[0]],[piron[1]],[piron[2]]]
    # inv_I=np.linalg.inv(I)
    # omegadotmat=dot(inv_I,mat_piron)
    # omegadot=[omegadotmat[0][0],omegadotmat[1][0],omegadotmat[2][0]]
    omegadot=[(tau[0]/I[0][0])-(((I[1][1]-I[2][2])/I[0][0])*omega[1][0]*omega[2][0]),(tau[1]/I[1][1])-(((I[2][2]-I[0][0])/I[1][1])*omega[0][0]*omega[2][0]),(tau[2]/I[2][2])-(((I[0][0]-I[1][1])/I[2][2])*omega[0][0]*omega[1][0])]
    return omegadot
    

        
    
def axe_x():
    axe_x=[]
    a=0
    for i in range(2001):
        axe_x.append([a+i*0.005])
    return(axe_x) 
         

    
    
    
def simulation():
    thetax=[]
    thetay=[]
    thetaz=[]
    xx=[]
    xy=[]
    xz=[]    
    omegax=[]
    omegay=[]
    omegaz=[]
    omegadotx=[]
    omegadoty=[]
    omegadotz=[]    
    thetadotx=[]
    thetadoty=[]
    thetadotz=[]    
    w1=[]
    w2=[]
    w3=[]
    w4=[]
        
    Z=axe_x()    

    KPpsi=10
    KPtheta=10
    KPphi=10
    KDpsi=1.75
    KDtheta=1.75
    KDphi=1.75
    KIpsi=5.5
    KItheta=5.5
    KIphi=5.5

    intpsi=0
    inttheta=0
    intphi=0

    i=0
    times=[]
    while i<=2000:
        times.append(i)
        i+=1
    N=len(times)
    
    
    x=[0,0,10]
    xdot=[0,0,0]
    theta=[10,10,10]
    thetamat=[[theta[0]],[theta[1]],[theta[2]]]
    deviation=100
    # thetadot=[deg2rad(2*deviation*(np.random.rand())-deviation),deg2rad(2*deviation*(np.random.rand())-deviation),deg2rad(2*deviation*(np.random.rand())-deviation)]
    thetadot=[0,0,0]
  
    
    
    
    
    
    
    
    
    for t in times:
        
        H=(m*g)/(4*k*cos(deg2rad(theta[1]))*cos(deg2rad(theta[0])))
        intpsi+=dt*theta[0]
        inttheta+=dt*theta[1]
        intphi+=dt*theta[2]
        upsi=KDpsi*thetadot[0]+KPpsi*theta[0]+KIpsi*intpsi
        utheta=KDtheta*thetadot[1]+KPtheta*theta[1]+KIpsi*inttheta
        uphi=KDphi*thetadot[2]+KPphi*theta[2]+KIphi*intphi
        
        i=[H-((2*b*upsi*I[0][0]+uphi*I[2][2]*k*L)/(4*b*k*L)),H+(uphi*I[2][2])/(4*b)-(utheta*I[1][1])/(2*k*L),H-((-2*b*upsi*I[0][0]+uphi*I[2][2]*k*L)/(4*b*k*L)),H+(uphi*I[2][2])/(4*b)+(utheta*I[1][1])/(2*k*L)]
        
        
        
        R1=thetadottoomega(thetamat)
        R2=omegatothetadot(thetamat)
        
        omegamat=dot(R1,[[thetadot[0]],[thetadot[1]],[thetadot[2]]])
        omega=[omegamat[0][0],omegamat[1][0],omegamat[2][0]]
        
        a=acceleration(i,thetamat,xdot,m,g,k,kd)
        omegadot=angular_acceleration(i,omegamat,I,L,b,k)

        omega=SommeListe([omega,mulmatscal(dt,omegadot)])
        omegamat=[[omega[0]],[omega[1]],[omega[2]]]
       
        thetadotmat=dot(R2,omegamat)
        
        thetadot=[thetadotmat[0][0],thetadotmat[1][0],thetadotmat[2][0]]
        theta=SommeListe([theta,mulmatscal(dt,thetadot)])
        thetamat=[[theta[0]],[theta[1]],[theta[2]]]
        
        xdot=SommeListe([xdot,mulmatscal(dt,a)])
        x=SommeListe([x,mulmatscal(dt,xdot)])
        
        
        
        
        thetax.append(theta[0])
        thetay.append(theta[1])
        thetaz.append(theta[2])
        xx.append(x[0])
        xy.append(x[1])
        xz.append(x[2])
        omegax.append(omega[0])
        omegay.append(omega[1])
        omegaz.append(omega[2])
        omegadotx.append(omegadot[0])
        omegadoty.append(omegadot[1])
        omegadotz.append(omegadot[2])
        thetadotx.append(thetadot[0])
        thetadoty.append(thetadot[1])
        thetadotz.append(thetadot[2])
        w1.append(i[0])
        w2.append(i[1])
        w3.append(i[2])
        w4.append(i[3])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    plot(Z,thetax,label="thetax")
    plot(Z,thetay,label="thetay")
    plot(Z,thetaz,label="thetaz")
    print(len(thetax))
    
    plot(Z,xx,label="xx")
    plot(Z,xy,label="xy")
    plot(Z,xz,label="xz")
    # 
    plot(Z,omegax,label="omegax")
    plot(Z,omegay,label="omegay")
    plot(Z,omegaz,label="omegaz")
    
    plot(Z,omegadotx,label="omegadotx")
    plot(Z,omegadoty,label="omegadoty")
    plot(Z,omegadotz,label="omegadotz")
    
    plot(Z,thetadotx,label="thetadotx")
    plot(Z,thetadoty,label="thetadoty")
    plot(Z,thetadotz,label="thetadotz")
    
    plot(Z,sqrt(w1),label="w1")
    plot(Z,sqrt(w2),label="w2")
    plot(Z,sqrt(w3),label="w3")
    plot(Z,sqrt(w4),label="w4")
    
    legend()
    show()
    
    # NomFichier='/Users/PB/Desktop/simulationcorrecteur'
    # Fichier=open(NomFichier,'w')
    # for y in range(2000):
    #     sthetax=str(thetax[y])
    #     sthetay=str(thetay[y])
    #     sthetaz=str(thetaz[y])
    #     sxx=str(xx[y])
    #     sxy=str(xy[y])
    #     sxz=str(xz[y])
    #     sw1=str(w1[y])
    #     sw2=str(w2[y])
    #     sw3=str(w3[y])
    #     sw4=str(w4[y])
    #     
    #     
    #     Fichier.write(sthetax)
    #     Fichier.write('\t')
    #     Fichier.write(sthetay)
    #     Fichier.write('\t')
    #     Fichier.write(sthetaz)
    #     Fichier.write('\t')
    #     Fichier.write(sxx)
    #     Fichier.write('\t')
    #     Fichier.write(sxy)
    #     Fichier.write('\t')
    #     Fichier.write(sxz)
    #     Fichier.write('\t')
    #     Fichier.write(sw1)
    #     Fichier.write('\t')
    #     Fichier.write(sw2)
    #     Fichier.write('\t')
    #     Fichier.write(sw3)
    #     Fichier.write('\t')
    #     Fichier.write(sw4)
    #     Fichier.write('\n')
    # Fichier.close()
