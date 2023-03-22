#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:02:28 2017

proyecto de grado

@author: franciscorealescastro
"""
import cv2
import numpy as np
import itertools
from scipy import ndimage
from skimage import morphology 
from time import time
import math as mt
from sklearn.mixture import GaussianMixture
def roundOdd(n):
    answer = round(n)
    if  answer%2:
        return answer
    if abs(answer+1-n) < abs(answer-1-n):
        return answer + 1
    else:
        return answer - 1


def datosInterp(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#se convierte a HSV
    V=hsv[:,:,2]#se retiene el canal V
    [fil,col,c]=img.shape
    mascara=np.zeros((fil,col))-1
    mascara[0:np.uint64(np.round(0.31*fil)),0:np.uint64(np.round(0.28*col))]=1
    mascara[0:np.uint64(np.round(0.31*fil)),np.uint64(np.round(0.72*col)):col]=1
    mascara[np.uint64(np.round(0.69*fil)):fil,0:np.uint64(np.round(0.28*col))]=1
    mascara[np.uint64(np.round(0.69*fil)):fil,np.uint64(np.round(0.72*col)):col]=1
    fraccionesV=mascara*V
    x=np.zeros(np.sum(mascara==1))
    y=np.zeros(np.sum(mascara==1))
    z=np.zeros(np.sum(mascara==1))
    c=0
    for i in range(0,fil):
        for j in range(0,col):
            if fraccionesV[i,j]>0:
               x[c]=j
               y[c]=i
               z[c]=fraccionesV[i,j]
               c=c+1
    return x,y,z

def obtenerInterp(x,y,z,img):

    #z=np.random.random(numdata)
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    [ny,nx,c] = img.shape
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)
    return xx,yy,zz

def polyfit2d(x, y, z, order=2):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z,rcond=None)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j    
    return z 
def elimSombras(img):
    x,y,z=datosInterp(img)
    xx,yy,zz=obtenerInterp(x,y,z,img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#se convierte a HSV
    Vorig=hsv[:,:,2]#se retiene el canal V
    Vproc=Vorig/zz
    Uorig=np.mean(Vorig)
    Uproc=np.mean(Vproc)
    Vnew=Vproc*(Uorig/Uproc)
    hsv[:,:,2]=Vnew
    imgProc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return imgProc

def getsamples(img):
    x, y, z = img.shape
    samples = np.empty([x * y, z])
    index = 0
    for i in range(x):
        for j in range(y):
            samples[index] = img[i, j]
            index += 1
    return samples

def EMSegmentation(img, no_of_clusters=2):
    output = img.copy()
    samples = getsamples(img)
    em = cv2.ml.EM_create()
    em.setClustersNumber(no_of_clusters)
    em.trainEM(samples)
    means = em.getMeans()
    covs = em.getCovs()
    media1=means[0,:]
    media2=means[1,:]
    Mvar1=covs[0]
    Mvar2=covs[1]
    k1=1/np.sqrt(np.linalg.det(Mvar1))
    k2=1/np.sqrt(np.linalg.det(Mvar2))
    inv1=np.linalg.inv(covs[0])
    inv2=np.linalg.inv(covs[1])
    x, y, z = img.shape
    imgd=np.double(img)
    mascaraEM=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            c=imgd[i,j,:]
            p1=k1*np.exp(-0.5*(c-media1).dot(inv1.dot(c-media1)))
            p2=k2*np.exp(-0.5*(c-media2).dot(inv2.dot(c-media2)))
            maximo=np.max([p1,p2])
            if maximo==p1:
               imgd[i,j,:]=media1
               mascaraEM[i,j]=1
            elif maximo==p2:
                imgd[i,j,:]=media2
                mascaraEM[i,j]=0
    if np.sum(mascaraEM)>np.sum(mascaraEM!=1):
        mascaraEM=1*(mascaraEM!=1)
        
    return mascaraEM

def segmentarLunar(img):
    imgProc=elimSombras(img)
    mascara= EMSegmentation(imgProc,2)

    output = cv2.connectedComponentsWithStats(np.uint8(255*mascara), 4, cv2.CV_32S)
    labels = output[1]
    stats = output[2]
    mascara=(np.argmax(stats[:,4][1:])+1==labels)
    mascara=ndimage.binary_fill_holes(mascara).astype(int)
    mascaraDil=np.uint8(mascara*255)
    Amax=np.sum(mascaraDil)/255
    return mascaraDil,Amax

def obtenerBordes(mascara,Amax):#obtiene los bordes a partir de la multiplicacion entre la dilatacion hacia fuera y hacia adentro
    sqA=np.sqrt(Amax)
    re=np.round(0.02*sqA)#0.0266
    ri=np.round(0.15*sqA)
    kernelE=morphology.disk(re)
    kernelI=morphology.disk(ri)
    mascaraDil=np.uint8(255*(cv2.dilate(mascara, kernelE, iterations=1)/255)*(cv2.dilate(np.uint8(255*(mascara==0)), kernelI, iterations=1)/255) )
    return mascaraDil
#caracteristicas de asimetria (todas reciben la mascara)
#     Ap area de la lesion
def AreaLesion(mascara):#calcula el area de la lesion a partir de la mascara [0,255]
    mascara=np.double(mascara)/255
    Ap=np.sum(mascara)
    return Ap#retorna el area de la lesion

#     Pp perimetro de la lesion     
def PerLesion(mascara):#calcula el perimetro de la lesion a apartir de la mascara
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    Pp = cv2.arcLength(cnt,True)
    return Pp#retorna el perimetro

#     Ac Area del convexHull y Pc perimetro del convexHull
def CaracConvexHull(mascara):#calcula el area y perimetro del convex hull a partir de la mascara 
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    puntosConvex=hull[:,0,:]
    m,n=mascara1.shape
    ar=np.zeros((m,n))
    mascaraConvex=cv2.fillConvexPoly(ar, puntosConvex, 1)#Mascara del Convex Hull
    Ac=np.sum(mascaraConvex)#Area del convex hull
    mascaraConvex1=np.uint8(mascaraConvex.copy())
    imC, contoursC, hierarchyC = cv2.findContours(mascaraConvex1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cntC = contoursC[0]
    Pc = cv2.arcLength(cntC,True)#Perimetro del convex hull
    return Ac,Pc#retorna el area y perimetro del convex hull

#   Ab  Area del bounding box y Pb perimetro del bounding box y W/h la tasa de aspecto 
def CaracBoundBox(mascara):#calcula el area, perimetro y la division entre los lados del bounding box a partir de la mascara
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    m,n=mascara1.shape
    ar=np.zeros((m,n))
    mascaraRect=cv2.fillConvexPoly(ar, box, 1)#Mascara del Bounding Box
    Ab=np.sum(mascaraRect)#Area de bounding box
    mascaraRect1=np.uint8(mascaraRect.copy())
    imR, contoursR, hierarchyR = cv2.findContours(mascaraRect1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cntR = contoursR[0]
    Pb = cv2.arcLength(cntR,True)#perimetro del bounding box
    centro,dimensiones,rotacion = cv2.minAreaRect(cntR)#centro, longitud de los lados y rotacion del bounding box
    tasaAspecto= float(dimensiones[1])/float(dimensiones[0]) if dimensiones[1]>dimensiones[0] else float(dimensiones[0])/float(dimensiones[1])# tasa de aspecto: division entre los lados del bounding box
    return Ab,Pb,tasaAspecto#retorna el area, perimetro y division entre los lados del bounding box

#   Ae area de la elipse Pe perimetro de la elipse MA eje mayor y ma eje menor
def CaracElipse(mascara):#Calcula Ae area de la elipse Pe perimetro de la elipse MA eje mayor y ma eje menor a partir de la mascara
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
    Ae=np.pi*MA*ma/4#area de la elipse
    Pe=np.pi*np.sqrt((MA**2+ma**2)/2)#perimetro elipse
    return Ae,Pe,MA,ma#retorna area de la elipse, perimetro de la elipse, eje mayor y eje menor

def fraccionar(mascara,n):#fracciona la mascara en n pedazos 
    vx,vy,x,y=orientacion(mascara)
    maskFrac=fracAngulo(mascara,vx,vy,x,y,n)
    return maskFrac

def fracAsim(mascara):#fracciona en 4 pedazos
    a=fraccionar(mascara,4)
    b=[]
    for i in a:
        b.append(mascara*i)
    return b

def caracAsimetria(mascara):# apartir de la amascara saca las caracterosticas de asimetria mascara [0,255]
    Ap=AreaLesion(mascara)#calcula el area de la lesion a partir de la mascara, mascara tiene que ser [0,255]
    Pp=PerLesion(mascara)#calcula el perimetro de la lesion a apartir de la mascar
    Ac,Pc=CaracConvexHull(mascara)#calcula el area y perimetro del convex hull a partir de la mascara
    Ab,Pb,tasaAspecto=CaracBoundBox(mascara)#calcula el area, perimetro y la division entre los lados del bounding box a partir de la mascara
    Ae,Pe,MA,ma=CaracElipse(mascara)#Calcula Ae area de la elipse Pe perimetro de la elipse MA eje mayor y ma eje menor a partir de la mascara
    A1=Ap/Ab #area lesion/area bounding box
    A2=Ac/Ab #area convex hull/area bounding box
    A3=Ap/Ac #area de la lesion/area del convex hull
    A4=np.sqrt(4*Ap/np.pi)/Pb#diametro equivalente/perimetro bounding box
    A5=4*np.pi*(Ap/(Pp**2))#circularidad 4*pi(Ap/Pp**2) da 1 si es una circunferencia perecta
    A6=Pp/Pb#perimetro de la lesion/perimetro bounding box
    A7=ma/MA#radio inferior elipse/radio inferior elipse
    A8=Pc/Pb#Perimetro convex hull/perimetro bounding box
    A9=tasaAspecto#tasa de aspecto bb/ab division de los lados del bounding box
    A10=Ap/Ae# Area de la lesion/Area de la elipse
    A11=Pp/Pe#perimetro de la lesion/perimetro de la elipse
    frac=fracAsim(mascara)# devuelve lista con la mascara dividida en 4 partes
    B1=np.double(frac[0]+frac[1])/255#suma dos cuadrantes en un eje
    B2=np.double(frac[2]+frac[3])/255#suma dos cuadrantes en un eje
    #A12=np.abs(np.sum(B1)-np.sum(B2))/np.sum(np.double(mascara)/255)#tasa de areas ap=(A1-A2)/Ap diferencia de areas de los pedazos cortados por el eje ap entre el area de la lesion 
    A13=np.sum(B2)/np.sum(B1) if np.sum(B1)>np.sum(B2) else np.sum(B1)/np.sum(B2)#tasa de forma ap=A1/A2 
    B1=np.double(frac[1]+frac[2])/255#suma los dos cuadrantes en el otro eje
    B2=np.double(frac[0]+frac[3])/255#suma los dos cuadrantes en el otro eje
    #A14=np.abs(np.sum(B1)-np.sum(B2))/np.sum(np.double(mascara)/255)#tasa de areas bp=(B1-B2)/Ap
    A15=np.sum(B2)/np.sum(B1) if np.sum(B1)>np.sum(B2) else np.sum(B1)/np.sum(B2)#tasa de forma bp=B1/B2
    A=np.array([A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A13,A15])
    
    return A

def textuVar(img, p):#a partir de la imagen a color original geera el canal de variacion de textura I1
    img1=np.double(img)/255#toma la imagen original a color y la convierte a [0,1]
    L=(img1[:,:,0]+img1[:,:,1]+img1[:,:,2])/3#promedia los canales (imagen en escala de grises)
    [m,n]=L.shape#tamano de la imagen
    Taumax=np.zeros((m,n))#se define el Tau maximo como una imagen con las mismas dimensiones
    desvMax=np.sqrt(m*n)*0.05737

    for c in range(0,p):#se calcula para nueve desviaciones diferentes
        desv= 1+(desvMax-1)*(c/p)#empieza en 1 y termina en 43/7
        tam=round(7*desv)#el tamano de la ventana empieza en 7 y termina en 43
        if tam % 2 == 0:
            tam=tam+1
        tam=np.uint64(tam)   
        S=cv2.GaussianBlur(L,(tam,tam),desv,0)#se filtra la imagen L con el filtro gaussiano de tamano tam y desviacion desv  
        Sn=1-S#complemento
        tau=L*(Sn)/S#nueva imagen de textura para la desviacion desv actual 
        for i in range(0,m):
            for j in range(0,n):
               if tau[i,j]>Taumax[i,j]:
                  Taumax[i,j]=tau[i,j] #se guarda el maxima textura Tau generadas entre las desviaciones 1,11/7,...,43/7
    i1=(Taumax-np.min(Taumax))/(np.max(Taumax)-np.min(Taumax))#se normaliza para que todos los valores queden entre 0 y 1
    return i1

def oscuInfo(img):#calcula la imagen de oscuridad I2 a partir de la imagen original 
    img1=np.double(img)/255#o pasa a [0,1]
    i2=(1-img1[:,:,0])#calcula el negativo de la componente en rojo
    return i2

def colInfo(img,mascara):#calcula el canal de informacion de color I3 a partir de la imagen original y su mascara
    mascara1=mascara.copy()#copia de la mascara
    mascara1=np.double(mascara1)/255#mascara con valores [0,1]
    a1=img[:,:,0]#canal rojo
    a2=img[:,:,1]#canal verde
    a3=img[:,:,2]#canal azul
    d1=a1.flatten()#pone todos los datos del canal rojo en un arreglo
    d2=a2.flatten()#pone todos los datos del canal verde en un arreglo
    d3=a3.flatten()#pone todos los datos del canal azul en un arreglo
    m=d1.shape#cantidad de datos
    m=m[0]#cantidad de datos
    mean=[np.mean(d1),np.mean(d2),np.mean(d3)]#arreglo con las medias de los valores de cada canal RGB 
    datos=np.array((d1,d2,d3))# matriz de 3xm resultante de concatenar todos los datos de los canales RGB
    datos=datos.T-mean#se le quita la media para que el promedio de cero
    cov=np.cov(datos.T)#matriz de covarianza
    valores, vectores = np.linalg.eigh(cov)#Calculo de los valores propios y vectores propios
    vectores=-vectores
    U=vectores.copy()#copia de los vectores propios
    U[:,2]=vectores[:,0]#ordena los vectores
    U[:,0]=vectores[:,2]
    u1=U[:,0]#calcula u1 que es el vector que apunta hacia la mayor varicion de color
    ic=np.double(img.copy())/255#imagen original entre 0 y 1
    c=np.double(a1.copy())#imagen con el mismo tamano de la imagen original
    [m,n,ca]=ic.shape
    for i in range(m):
        for j in range(n):
            ic[i,j]=ic[i,j]-mean #a cada valor de la imagen original se le quita la media
            c[i,j]=np.abs(np.dot(u1,ic[i,j]))#la variable c va a almacenar el producto punto entre u1 y la imagen de color sin media 
            i3=(c-np.min(c))/(np.max(c)-np.min(c))#se normaliza [0,1]
    a=np.sqrt(np.sum(mascara1))        
    k=np.uint64(roundOdd(0.0735*a))
    I3=cv2.medianBlur(np.uint8(255*i3),k)#se filtra el ruido de color    
    I3=np.double(I3)/255    
    return I3

def varIm(I3seg,mascara):#calcula la varianza de los datos de la imagen I3seg contenidos en la mascara 
    mascara1=mascara.copy()#copia mascara
    meanI3=np.sum(I3seg)/np.sum(mascara1)#media I3
    I3segCentrada=(I3seg-meanI3)*mascara1#imagen con media cero
    I3segCentrada=I3segCentrada**2#cuadrado de los valores de la imagen
    varI3=np.sum(I3segCentrada)/np.sum(mascara1)#varianza de los valores de la imagen que estan contenidos en la mascara
    return varI3

def fracBordes(I):#lista de 8 fracciones de la imagen I
    a=fraccionar(I,8)
    b=[]
    for i in a:
        b.append(I*i)
    return b   

def caraColor(img,mascara,I3):#calculo de las caracteristicas de color a partir de la imagen original [0,255] y la mascara [0,255]
    mascara1=mascara.copy()#copia de la mascara
    mascara1=np.double(mascara1)/255#mascara entre 0 y 1
    I3Seg=I3*mascara1

    valoresI3=I3Seg[mascara1>0].flatten()
    histColor=caracHist(valoresI3,bins=10)
    

    ic=np.double(img)/255#imagen a color [0,1]
    icSeg=ic.copy()
    #Imagen original segmentada
    icSeg[:,:,0]=ic[:,:,0]*mascara1
    icSeg[:,:,1]=ic[:,:,1]*mascara1
    icSeg[:,:,2]=ic[:,:,2]*mascara1
    a=np.sqrt(np.sum(mascara1))#longitud equivalente
    k=0.0245#constante que pasa de la lingitud equivalente a la desviacion del filtro gaussiano (lo hace invariante con el tamano)
    ic1=cv2.GaussianBlur(icSeg,(0,0),k*a,0)#filtro gaussiano con desviacion proporcional al tamano de la imagen  
    
    cont=np.zeros((6,1))
    cont=cont.flatten()#creacion de 6 contadores
    #definicion de los colores de interes
    blanco=np.array([1,1,1])#Blanco
    red=np.array([0.8,0.2,0.2])#Rojo
    cafeC=np.array([0.6,0.4,0])#Cafe claro
    cafeO=np.array([0.2,0,0])#cafe oscuro
    grisAzul=np.array([0.2,0.6,0.6])#gris azulado
    
    [m,n,c]=ic.shape
    # colCont=icSeg.copy()

    for i in range(0,m):
        for j in range(0,n):
            if mascara1[i,j]==1:#pixeles que pertenecen a la lesion
                c=np.array([np.linalg.norm(ic1[i,j]-blanco),np.linalg.norm(ic1[i,j]-red),np.linalg.norm(ic1[i,j]-cafeC),np.linalg.norm(ic1[i,j]-cafeO),np.linalg.norm(ic1[i,j]-grisAzul),np.linalg.norm(ic1[i,j])])

                d=(c==np.min(c))#1 en la minimo valor de Idist para el pixel i,j 
                cont+=d#suma a cada una de las 6 posiciones si la distancia correspondiente es minima
                # if d[0]==True:
                #     colCont[i,j]=np.array([1,1,1])
                # if d[1]==True:
                #     colCont[i,j]=np.array([0.2,0.2,0.8])
                # if d[2]==True:
                #     colCont[i,j]=np.array([0,0.4,0.6])
                # if d[3]==True:
                #     colCont[i,j]=np.array([0,0,0.2])
                # if d[4]==True:
                #     colCont[i,j]=np.array([0.6,0.6,0.2])
                # if d[5]==True:
                #     colCont[i,j]=np.array([0,0,0])   
     
    cont=cont/np.sum(mascara1)#se normalizan los datos para que la suma de los contadores de 1 
    # colCont[:,:,0]=colCont[:,:,0]+1*(mascara1==0)
    # colCont[:,:,1]=colCont[:,:,1]+1*(mascara1==0)
    # colCont[:,:,2]=colCont[:,:,2]+1*(mascara1==0)
    C=np.concatenate((histColor,cont),axis=0)
    return C
def orientacion(mascara):
    mascara1=mascara.copy()
    im2, contours, hierarchy = cv2.findContours(mascara1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    return vx,vy,x,y


def fracAngulo(mascara,vx,vy,x,y,n):
    rows,cols = mascara.shape[:2]
    fracciones=[]
    a1=np.zeros((rows,cols))
    a2=np.zeros((rows,cols)) 
    k=90*(4/n)  
    t=k*(np.pi/180)
    for cont in range(0,n):
        for i in range(0,rows):
            for j in range(0,cols):
                a1[i,j]=np.dot([i,j],[-vx[0],vy[0]])-np.dot([y[0],x[0]],[-vx[0],vy[0]])>0      
                
        [vx1,vy1]=[vx*np.cos(t)-vy*np.sin(t),vx*np.sin(t)+vy*np.cos(t)]
        
        if np.dot([vx1[0],vy1[0]],[vx[0],vy[0]])<0:
            [vx1[0],vy1[0]]=[-vx1[0],-vy1[0]]
        
        for i in range(0,rows):
            for j in range(0,cols):
                a2[i,j]=np.dot([i,j],[-vx1[0],vy1[0]])-np.dot([y[0],x[0]],[-vx1[0],vy1[0]])>0 
       
        vx=vx1
        vy=vy1
        af1=(a1-a2)>0
    
        fracciones.append(af1)
    return fracciones

def caracHist(datos,bins=10):
    datosNorm=(datos-np.min(datos))/(np.max(datos)-np.min(datos))
    
    contBins=[0]*bins
    
    posBins=(datosNorm*(bins-1))+1
    
    for i in posBins:
        contBins[mt.floor(i)-1]+=1-(i-mt.floor(i))
        contBins[mt.ceil(i)-1]+=1-(mt.ceil(i)-i)
    
    caracteristicas=contBins/np.linalg.norm(contBins)
    return caracteristicas

def bordesMaximo(mask,dI3B):
    M = cv2.moments(mask)
    
    cX = int(M["m10"] / M["m00"])
    
    cY = int(M["m01"] / M["m00"])
    
    value = np.sqrt(((cX)**2.0)+((cY)**2.0))
    
    polar_image = cv2.linearPolar(dI3B,(cX, cY), value, cv2.WARP_FILL_OUTLIERS)

    m,n=polar_image.shape
    maximo=np.zeros((m,n))
    for i in range(0,m):
        for j in range(0,n):
            if polar_image[i,j]==np.max(polar_image[i,:]):
               maximo[i,j]=1 
               
    maximoC = cv2.linearPolar(maximo,(cX, cY), value,cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    return maximoC    
def caracBordes(img,mascara,I,I3):
    B=np.double(I)/255#pasa la imagen I que son los bordes del lunar al intervalo[0,1]
    mascara1=mascara.copy()#copia de la mascara
    m,n=mascara1.shape#tamano de la mascara

    I3dx = cv2.Sobel(I3,cv2.CV_64F,1,0)#calculo de la componente en x del gradiente para el canal de informacion de color
    I3dy = cv2.Sobel(I3,cv2.CV_64F,0,1)#calculo de la componente en y del gradiente para el canal de informacion de color
    dI3=np.sqrt(I3dx**2+I3dy**2)#magnitode del gradiente
    dI3B=dI3*B#valores del gradiente que pertenecen a los bordes
    bordes=bordesMaximo(mascara, dI3B)

    fraccionesB=fracBordes(I)#calcula lista con 8 trozos de los bordes I
    
    V=[]
    gB3=[]

    for i in fraccionesB:
        gB3.append(np.double(i)*bordes*dI3B/255)#gB3 es la lista que guarda una imagen con los valores de la magnitud del gradiente de I3 en cada porcion de los 8 bordes    
        V.append(np.sum(np.double(i)*bordes*dI3B/255)/np.sum(np.double(i)*bordes/255))

    B1=np.sort(V)
    
    datos=dI3[bordes>0]
    B2=caracHist(datos,bins=10)
    B=np.concatenate((B1,B2))
    return B

def caracDifEstruct(img,mascara):#calculo de las caracteristicas de textura a partir de la imagen original y la mascara
    i1=textuVar(img, p=10)
    I1dx = cv2.Sobel(i1,cv2.CV_64F,1,0)#calculo de la componente en x del gradiente para el canal de informacion de color
    I1dy = cv2.Sobel(i1,cv2.CV_64F,0,1)#calculo de la componente en y del gradiente para el canal de informacion de color
    dI1=np.sqrt(I1dx**2+I1dy**2)
    Amask=np.sum(mascara)/255
    tam=round(11*(Amask/7796))
    if tam % 2 == 0:
        tam=tam+1
        
    desv=tam/7
    tam=np.uint64(tam)
    dI1SegG=cv2.GaussianBlur(dI1,(tam,tam),desv,0)*mascara/255
    
    
    datosDif=dI1SegG[mascara>0].flatten()
    D=caracHist(datosDif)
    return D
def removeHair(img):

    
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    m,n=np.shape(grayScale)
    
    tamK=round((5/325)*np.sqrt(m*n))
    if tamK % 2 ==0:
        tamK+=1
    tamK=np.uint64(tamK)    
    
    kernel = cv2.getStructuringElement(1,(tamK,tamK))
    
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    imgN = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)

    return imgN
#pendientes: 
#el canal de informacion de textura I1 hay que hacerlo independiente de la resolucion 
# Los canales I1, I2, I3 realmente son buenos para sacar caracteristicas de textura y bordes si estan normalizados entre 0 y 1?       
# cabiar c17,18 y 19 de varianzas a medias    
# revisar D1 y D2 ya que puede que pase que siempre de 1 y 0


    
#--------------------------------CARACTERISTICAS-----------------------------------------------
#-------------------------------- Asimetria -----------------------------------------------
#    A1=Ap/Ab area lesion/area bounding box
#    A2=Ac/Ab area convex hull/area bounding box
#    A3=Ap/Ac area de la lesion/area del convex hull
#    A4=np.sqrt(4*Ap/np.pi)/Pb#diametro equivalente/perimetro bounding box
#    A5=4*np.pi*(Ap/(Pp**2)) circularidad da 1 si es una circunferencia perecta
#    A6=Pp/Pb perimetro de la lesion/perimetro bounding box
#    A7=ma/MA radio inferior elipse/radio inferior elipse
#    A8=Pc/Pb Perimetro convex hull/perimetro bounding box
#    A9=tasa de aspecto bb/ab division de los lados del bounding box
#    A10=Ap/Ae Area de la lesion/Area de la elipse
#    A11=Pp/Pe perimetro de la lesion/perimetro de la elipse
#    A12(ELIMINADA)=tasa de areas ap=(A1-A2)/Ap diferencia de areas de los pedazos cortados por el eje ap entre el area de la lesion 
#    A13=tasa de forma ap=A1/A2 
#    A14(ELIMINADA)=tasa de areas bp=(B1-B2)/Ap
#    A15=tasa de forma bp=B1/B2
#-------------------------------- Bordes -----------------------------------------------
#    B1=media de los valores del gradiente de I1 que pertenecen al borde
#    B2=media de los valores del gradiente de I2 que pertenecen al borde
#    B3=media de los valores del gradiente de I3 que pertenecen al borde 
#    B4=varianza de los valores del gradiente de I1 en los bordes
#    B5=varianza de los valores del gradiente de I2 en los bordes
#    B6=varianza de los valores del gradiente de I3 en los bordes   
#    B7=promedio de medias del gradiente en cada una de las 8 porciones para I1 
#    B8=varianza de las medias de las 8 porciones para I1      
#    B9=promedio de medias del gradiente en cada una de las 8 porciones para I2  
#    B10=varianza de las medias de las 8 porciones para I2    
#    B11=promedio de medias del gradiente en cada una de las 8 porciones para I3 
#    B12=varianza de las medias de las 8 porciones para I3
#-------------------------------- Color -----------------------------------------------
#    c1=maximo del canal de informacion de color I3 que pertnece al lunar
#    c2=minimo del canal de informacion de color I3 que pertnece al lunar
#    c3=media de los valores del canal de informacion de color I3
#    c4=varianza de los valores del canal de informacion de color I3
#    c5=maximo del canal R de la imagen original Ic
#    c6=maximo del canal G de la imagen original Ic
#    c7=maximo del canal B de la imagen original Ic
#    c8=minimo del canal R de la imagen original Ic
#    c9=minimo del canal G de la imagen original Ic
#    c10=minimo del canal B de la imagen original Ic
#    c11=media del canal R de la imagen original Ic
#    c12=media del canal G de la imagen original Ic
#    c13=media del canal B de la imagen original Ic
#    c14=varianza del canal R de la imagen original Ic
#    c15=varianza del canal G de la imagen original Ic
#    c16=varianza del canal B de la imagen original Ic
#    c17=varianza del canal R / varianza del canal G (tenian que ser las medias)
#    c18=varianza del canal R / varianza del canal B
#    c19=varianza del canal G / varianza del canal B
#    c20=contador blanco        
#    c21=contador rojo  
#    c22=contador cafe claro
#    c23=contador cafe oscuro  
#    c24=contador gris azul  
#    c25=contador negro  
#    c26=media de las distancias con blanco
#    c27=media de las distancias con rojo
#    c28=media de las distancias con cafe claro
#    c29=media de las distancias con cafe oscuro
#    c30=media de las distancias con gris azulado
#    c31=media de las distancias con negro
#    c32=varianza de las distancias con blanco
#    c33=varianza de las distancias con rojo
#    c34=varianza de las distancias con cafe claro
#    c35=varianza de las distancias con cafe oscuro
#    c36=varianza de las distancias con gris azulado
#    c37=varianza de las distancias con negro     
#    c38=promedio de los valores de las 8 fracciones 
#    c39=varianza de los valores de las 8 fracciones    
#-------------------------------- Textura -----------------------------------------------
#    D1=maximo de los valores de textura que hace parte del lunar
#    D2=minimo de los valores de textura del lunar
#    D3=media de los valores de textura
#    D4=varianza de los valores de textura     
#for i in range(0,7):
    
#imagen="r"+str(15*i)+".jpg"
#imagen="r0.jpg"
# datos=[]
# etiquetas=[]
# mascaras=[]
# nombres=[]
# for i in range (1,32):
    
#     imagen="Bsimetrico"+str(i)+".jpg"
#     img = cv2.imread(imagen)
#     if img is not None: 
#         mask,Amax=segmentarLunar(img)
#         mascaras.append(mask)
#         nombres.append(imagen)
#         print("Listo: "+imagen)
        
#     imagen="Basimetrico"+str(i)+".jpg"
#     img = cv2.imread(imagen)
#     if img is not None:
#         mask,Amax=segmentarLunar(img)
#         mascaras.append(mask)
#         nombres.append(imagen)
#         print("Listo: "+imagen)
    
# datos=np.array(datos)
# etiquetas=np.array(etiquetas)    
inicio=time()    
imagen="Bsimetrico21.jpg"
img = cv2.imread(imagen)
#imgN=removeHair(img)


mask,Amax=segmentarLunar(img)

mascaraB=obtenerBordes(mask,Amax) 
I3=colInfo(img, mask)

# grad=cv2.imread("grad.jpg")
# valoresColor=grad[:,200,:]
# m,n=valoresColor.shape
# minV=np.min(V)
# dV=np.max(V)-np.min(V)
# colores=valoresColor[np.uint64(np.round(((V-minV)/dV)*(m-1))),:]

# r=np.ones((5,5))
# bordesColor=[]
# a=img.copy()
# img2=img.copy()
# dI=img.copy()
# dI[:,:,0]=255*dI3B
# dI[:,:,1]=255*dI3B
# dI[:,:,2]=255*dI3B
# for i in range(0,8):
#     b=(cv2.dilate(fraccionesB[i],r)-fraccionesB[i])/255
#     a[:,:,0]=(b)*colores[i,0]
#     a[:,:,1]=(b)*colores[i,1]
#     a[:,:,2]=(b)*colores[i,2]
#     m,n,_=a.shape
#     for x in range(0,m):
#         for y in range(0,n):
#             if b[x,y]!=0:
#                 img2[x,y,0]=colores[i,0]
#                 img2[x,y,1]=colores[i,1]
#                 img2[x,y,2]=colores[i,2]
                
#                 dI[x,y,0]=colores[i,0]
#                 dI[x,y,1]=colores[i,1]
#                 dI[x,y,2]=colores[i,2]



    
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
A=caracAsimetria(mask)
B=caracBordes(img, mask, mascaraB,I3)
C=caraColor(img,mask,I3)
D=caracDifEstruct(img, mask)

fin=time()
print("Duracion: "+str(fin-inicio)+" segundos")


    
