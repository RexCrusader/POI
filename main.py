from scipy.stats import norm
from csv import writer
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def generacja_pozioma(liczba_punktow):
    dystrybucja_x = norm(loc=0, scale=20)
    dystrybucja_y = norm(loc=0, scale=200)
    dystrybucja_z = norm(loc=0.2, scale=0.05)

    x = dystrybucja_x.rvs(size=liczba_punktow)
    y = dystrybucja_y.rvs(size=liczba_punktow)
    z = dystrybucja_z.rvs(size=liczba_punktow)

    punkty = zip(x, y, z)
    with open('Lab1DataPozioma.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in punkty:
            csvwriter.writerow(p)

def generacja_pionowa(liczba_punktow):
    dystrybucja_x = norm(loc=0, scale=20)
    dystrybucja_y = norm(loc=0.2, scale=0.05)
    dystrybucja_z = norm(loc=0, scale=200)

    x = dystrybucja_x.rvs(size=liczba_punktow)
    y = dystrybucja_y.rvs(size=liczba_punktow)
    z = dystrybucja_z.rvs(size=liczba_punktow)

    punkty = zip(x, y, z)
    with open('Lab1DataPionowa.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in punkty:
            csvwriter.writerow(p)

def generacja_cylidryczna(liczba_punktow):
    kat = np.pi
    dystrybucja_x = norm(loc=0, scale=kat)
    dystrybucja_y = norm(loc=0.5, scale=20)
    dystrybucja_z = norm(loc=0, scale=kat)

    x = dystrybucja_x.rvs(size=liczba_punktow)
    y = dystrybucja_y.rvs(size=liczba_punktow)
    z = dystrybucja_z.rvs(size=liczba_punktow)

    punkty = zip(x, y, z)
    with open('Lab1DataCylider.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in punkty:
            csvwriter.writerow(p)


def wczytaj_punkty(plik):
    punkty = []
    with open(plik, 'r', encoding='utf-8') as file:
        for line in file:
            x, y, z = map(float, line.strip().split(','))
            punkty.append((x, y, z))
    return np.array(punkty)




#generacja_pozioma(2000)
#generacja_pionowa(2000)
#generacja_cylidryczna(2000)

