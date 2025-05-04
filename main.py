from scipy.stats import norm
from csv import writer
import numpy as np
import random
from sklearn.cluster import KMeans


def generacja_pozioma(liczba_punktow):
    dystrybucja_x = norm(loc=0, scale=100)
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
    dystrybucja_x = norm(loc=0, scale=100)
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
    kat = np.linspace(0, 2 * np.pi, liczba_punktow, endpoint=False)
    x = 50 * np.cos(kat)
    dystrybucja_y = norm(loc=0.5, scale=50)
    z = 50 * np.sin(kat)

    y = dystrybucja_y.rvs(size=liczba_punktow)

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

def dopasuj_płaszczyznę_ransac(punkty, iteracje, threshold):
    najwięcej_inlierów = 0
    najlepsze_inliery = None

    for _ in range(iteracje):
        idx = random.sample(range(len(punkty)), 3)
        A, B, C = punkty[idx[0]], punkty[idx[1]], punkty[idx[2]]

        vec_va = A - C
        vec_vb = B - C

        vec_ua = vec_va / np.linalg.norm(vec_va)
        vec_ub = vec_vb / np.linalg.norm(vec_vb)
        vec_uc = np.cross(vec_ua, vec_ub)
        if np.linalg.norm(vec_uc) == 0:
            continue

        W = vec_uc / np.linalg.norm(vec_uc)
        D = -np.sum(np.multiply(W, C))

        odległości = np.abs(np.dot(punkty, W) + D) / np.linalg.norm(W)
        inliers = np.where(odległości <= threshold)[0]
        if len(inliers) > najwięcej_inlierów:
            najwięcej_inlierów = len(inliers)
            najlepsze_inliery = inliers

    if najlepsze_inliery is None or len(najlepsze_inliery) < 3:
        raise ValueError("Nie znaleziono wystarczającej liczby inlierów.")

    inlier_points = punkty[najlepsze_inliery]
    centroid = np.mean(inlier_points, axis=0)
    przesunięte = inlier_points - centroid
    _, _, Vt = np.linalg.svd(przesunięte)
    W_dokładne = Vt[-1]
    D_dokładne = -np.dot(W_dokładne, centroid)

    return W_dokładne, D_dokładne, inlier_points


def znajdz_klastry_kmeans(punkty, k):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    etykiety = kmeans.fit_predict(punkty)

    klastry = [punkty[etykiety == i] for i in range(k)]
    return klastry

def analiza_chmur(chmury, prog_płaszczyzny):
    for idx, punkty in enumerate(chmury):
        W, D, inliery = dopasuj_płaszczyznę_ransac(punkty,200,0.05)


        odległości = np.abs(np.dot(punkty, W) + D) / np.linalg.norm(W)
        średnia_odległość = np.mean(odległości)
        print("-------------------------------------------------------------------")
        print(f"\n[Chmura {idx + 1}]")
        print(f"- Średnia odległość do płaszczyzny: {średnia_odległość:.5f}")

        if średnia_odległość > prog_płaszczyzny:
            print("- NIE uznana za płaszczyznę (za duża średnia odległość)")
            continue

        print("- Jest uznana za płaszczyznę.")
        print("Wektor normalny:", W)
        print("D:", D)
        print("Liczba inlierów:", len(inliery))

        abs_w = np.abs(W / np.linalg.norm(W))
        if abs_w[2] > 0.9:
            print("- Orientacja: POZIOMA (norma zbliżona do osi Z)")
        elif abs_w[0] > 0.9 or abs_w[1] > 0.9:
            print("- Orientacja: PIONOWA (norma zbliżona do osi X lub Y)")

        print("-------------------------------------------------------------------")


#generacja_pozioma(2000)
#generacja_pionowa(2000)
#generacja_cylidryczna(2000)

punkty_poziome = wczytaj_punkty('Lab1DataPozioma.xyz')
punkty_pionowe = wczytaj_punkty('Lab1DataPionowa.xyz')
punkty_cylinder = wczytaj_punkty('Lab1DataCylider.xyz')

klastry_poziome = znajdz_klastry_kmeans(punkty_poziome, k=3)
klastry_pionowe = znajdz_klastry_kmeans(punkty_pionowe, k=3)
klastry_cylinder = znajdz_klastry_kmeans(punkty_cylinder, k=3)

print("\nAnaliza chmur dla wygenerowaniej chmury poziomej: ")
analiza_chmur(klastry_poziome,0.05)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("Analiza chmur dla wygenerowaniej chmury pionowej: ")
analiza_chmur(klastry_pionowe,0.05)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("Analiza chmur dla wygenerowaniej chmury cylindrycznej: ")
analiza_chmur(klastry_cylinder,0.05)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")