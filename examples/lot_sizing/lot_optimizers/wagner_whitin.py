#!/bin/bash
#!/usr/bin/env  python
# -*- coding: utf-8 -*-

import numpy as np

class WagnerWhitin():
    def __init__(self):
        pass

    def run(self, dict_in, demand):
        #Variablen definieren
        varKosten = dict_in["varCost"]
        fixKosten = dict_in["fixedCost"]
        bestellung = np.array(demand)   #Liste von den Bestellungen
        anzPeriode = len(demand)        #Anzahl der Periode
        minKosten = 0                     #Minimale Kosten
        kosten = [[0 for i in range(0, anzPeriode)] for j in range(0, anzPeriode)] #Kostenmatrix
        minListe = [0 for i in range(0, anzPeriode)] #liste von den minimalen Kosten je Periode
        periodeListe = []    #liste von den Bestellperioden
        bestellteMenge = [0 for i in range(0, anzPeriode)]  #Liste von den bestellten Mengen je Bestellperiode
        nullbestellung = 0  #Index, an welchen die Bestellungen null sind
        i = 0

        #aussuchen der Null Bestellungen ganz am Anfang
        #beginnen mit der ersten Nicht-Null Bestellung
        while bestellung[i] == 0:
            bestellteMenge[i] = 0
            i = i + 1
            if i == anzPeriode:
                i = i - 1
                break

        #Beginn der Schleife von Kostenmatrix
        for i in range(0, anzPeriode):
            if bestellung[i] > 0 or i == (anzPeriode - 1): #Bedingung erfüllt, wenn die Bestellmenge nicht null ist oder ganz am Ende der Bestellperiode
                for j in range(i, anzPeriode):
                    kosten[j][i] += fixKosten + minKosten
                    for k in range(i + 1, j + 1):
                        kosten[j][i] += varKosten * (k - i) * bestellung[k]
                minKosten = min(filter(None, kosten[i]))
                minListe[i] = minKosten
            else:
                i += 1

        #Rückwarts Schleife, Aussuchen der Bestellperiode,
        #mit der Beachtung, dass es Null Bestellungen dazwischen geben
        periode = anzPeriode
        while periode != 0:
            if bestellung[periode-1] != 0: #Wenn die Bestellung von vorherigen Periode gleich null ist, dann geht es zu vor vorherigen Periode
                index = kosten[periode-1].index(minListe[periode-1])
                periodeListe.insert(0, index)
                periode = index
                if periode == 0:
                    break
            else:
                periode = periode - 1

        #Berechnung der bestellten Mengen je Bestellperiode
        anzBestellperiode = len(periodeListe)
        j = 0
        for i in range (0, anzBestellperiode):
            j = periodeListe[i]
            if i < (anzBestellperiode-1):
                for j in range(j, periodeListe[i+1]):
                    bestellteMenge[periodeListe[i]] += bestellung[j]
            else:
                for j in range(j, anzPeriode):
                    bestellteMenge[periodeListe[i]] += bestellung[j]

        #Berechnung Bestellperiode + 1
        #for i in range(0, len(periodeListe)):
        #    periodeListe[i] = periodeListe[i] + 1

        #Definieren der Output Array
        if len(demand) == 0 or bestellung.max() == 0:
            output = np.array([0])
        else:
            #output = {"total_cost": np.array(kosten),
            #          "minimal": minListe,
            #          "Bestellperiode": periodeListe,
            #          "bestellte Mengen": bestellteMenge,
            #          "minimale Kosten": minKosten}
          #  print("Minimale Kosten: " + str(minKosten))
            output = np.array(bestellteMenge)
        return output


if __name__ == "__main__":
    # static test dictionary
    dict_in = {
        "planningPeriod": 30,  # Anzahl der Perioden
        "fixedCost": 490,  # Bestellkosten
        "varCost": 0.01,  # Lagerhaltungssatz
        "roll": 4
    }
    demand = [3057 ,	2860 ,	3033 ,	3182 ,	3179 ,	3118 ,	3159 ,	2930 ,	2857 ,	2962 ,	2874 ,	3134 ,	3104 ,	2938 ,	2746 ,	3073 ,	2909 ,	2789 ,	2826 ,	2999 ,	2848 ,	2719 ,	2851 ,	2733 ,	3074 ,	2773 ,	2902 ,	2998 ,	2959 ,	2842 ,	2889 ,	2718 ,	2940 ,	2855 ,	2799 ,	2896 ,	2805 ,	2878 ,	2770 ,	2788 ,	2947 ,	2982 ,	2687 ,	2750 ,	2823 ,	2647 ,	2722 ,	2918 ,	2722 ,	2839 ,	2914 ,	2702 ,	2763 ,	2795 ,	2792 ,	2573 ,	2870 ,	2847 ,	2733 ,	2746 ,	2525 ,	2710 ,	2528 ,	2510 ,	2651 ,	2620 ,	2790 ,	2621 ,	2745 ,	2553 ,	2807 ,	2427 ,	2712 ,	2436 ,	2499 ,	2748 ,	2785 ,	2543 ,	2604 ,	2436 ,	2374 ,	2443 ,	2632 ,	2708 ,	2407 ,	2578 ,	2595 ,	2358 ,	2659 ,	2693 ,	2654 ,	2328 ,	2637 ,	2302 ,	2658 ,	2532 ,	2554 ,	2322 ,	2357 ,	2330 ,	2479 ,	2387 ,	2543 ,	2331 ,	2322 ,	2342 ,	2569 ,	2498 ,	2357 ,	2236 ,	2383 ,	2201 ,	2214 ,	2555 ,	2310 ,	2371 ,	2243 ,	2414 ,	2511 ,	2231 ,	2360 ,	2527 ,	2492 ,	2356 ,	2400 ,	2314 ,	2426 ,	2201 ,	2363 ,	2461 ,	2206 ,	2248 ,	2320 ,	2096 ,	2162 ,	2370 ,	2284 ,	2131 ,	2326 ,	2073 ,	2314 ,	2233 ,	2146 ,	2232 ,	2368 ,	2384 ,	2181 ,	2293 ,	2249 ,	2032 ,	2099 ,	2023 ,	2059 ,	2199 ,	2322 ,	2302 ,	2067 ,	2116 ,	2066 ,	1955 ,	1936 ,	2279 ,	2315 ,	2032 ,	2147 ,	2088 ,	2147 ,	2057 ,	2277 ,	1964 ,	2036 ,	2161 ,	1877 ,	1897 ,	2023 ,	1962 ,	2081 ,	1986 ,	2166 ,	1852 ,	1898 ,	1868 ,	2206 ,	2003 ,	1914 ,	1988 ,	1816 ,	2047 ,	1871 ,	2144 ,	1929 ,	1898 ,	1865 ,	1871 ,	1741 ,	1801 ,	1920 ,	1750 ,	1785 ,	2077 ,	2016 ,	1722 ,	1846 ,	1975 ,	2026 ,	1868 ,	1909 ,	1915 ,	1819 ,	2052 ,	1701 ,	1958 ,	1800 ,	1829 ,	2030 ,	1816 ,	1663 ,	1743 ,	1672 ,	1698 ,	1673 ,	1705 ,	1601 ,	1938 ,	1971 ,	1881 ,	1827 ,	1726 ,	1732 ,	1690 ,	1621 ,	1651 ,	1824 ,	1562 ,	1856 ,	1699 ,	1662 ,	1634 ,	1882 ,	1626 ,	1505 ,	1489 ,	1722 ,	1729 ,	1693 ,	1722 ,	1651 ,	1587 ,	1621 ,	1518 ,	1719 ,	1443 ,	1493 ,	1742 ,	1732 ,	1680 ,	1428 ,	1447 ,	1674 ,	1553 ,	1718 ,	1662 ,	1503 ,	1707 ,	1408 ,	1647 ,	1549 ,	1719 ,	1667 ,	1590 ,	1377 ,	1654 ,	1319 ,	1462 ,	1672 ,	1623 ,	1461 ,	1286 ,	1597 ,	1359 ,	1460 ,	1432 ,	1407 ,	1557 ,	1321 ,	1626 ,	1522 ,	1535 ,	1346 ,	1305 ,	1506 ,	1221 ,	1256 ,	1583 ,	1455 ,	1536 ,	1361 ,	1300 ,	1450 ,	1535 ,	1526 ,	1446 ,	1393 ,	1259 ,	1226 ,	1228 ,	1420 ,	1377 ,	1309 ,	1471 ,	1289 ,	1171 ,	1151 ,	1212 ,	1157 ,	1208 ,	1376 ,	1323 ,	1234 ,	1352 ,	1437 ,	1200 ,	1073 ,	1258 ,	1162 ,	1087 ,	1094 ,	1321 ,	1183 ,	1022 ,	1005 ,	1031 ,	993 ,	1237 ,	1167 ,	1272 ,	1116 ,	1299 ,	1278 ,	1139 ,	1306 ,	1210 ,	1087 ,	972 ,	1014 ,	1028 ,	963 ,	1035 ,	896 ,	971 ,	1028 ,	984 ,	1249 ,	1098 ,	1114 ,	1101 ,	928 ,	926 ,	1147 ,	1073 ,	1130 ,	986 ,	889 ,	811 ,	970 ,	1000 ,	983 ,	976 ,	965 ,	994]
    #demand = [40, 50, 10, 20, 30, 40, 20, 25]
    #demand = [0, 0, 0, 10, 0, 1 ]
    #demand = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    #demand = [20, 50, 10, 50, 50, 10, 20, 40, 20, 30]
    #demand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    #demand = [0, 0, 1]
    #demand = [854, 1021, 984, 1030, 1240, 1178, 905, 1005, 958, 905, 966, 965, 1030, 1111, 1285, 1089, 959, 920]
    #demand = [300, 5001, 6022, 6103, 3533, 4046, 3044, 3023, 4064, 9552, 3960, 5077, 4000, 4000, 3430, 300, 3400, 3440, 200,
    #          3400, 9400, 4340, 3400, 300, 3040, 4000, 5000, 6500, 45454, 4443, 3244, 2334, 344, 3223, 2999, 4000, 4000,
    #          4000, 3400, 0, 0, 0, 4500, 400, 4500, 3400, 5400, 3000, 600, 0, 4500, 0, 0, 555, 4540, 500, 800,
    #          4555, 3000, 4555, 455, 3444, 4333, 2344, 4454, 4555, 3444]
    # demand = [0, 0, 6022, 6103, 3533, 4046, 3044, 3023, 4064, 9552, 3960, 5077, 4000, 4000, 3430, 300, 3400, 3440, 200,
    #           3400, 9400, 4340, 3400, 300, 3040, 4000, 5000, 6500, 45454, 4443, 3244, 2334, 344, 3223, 2999, 4000, 4000,
    #           4000, 3400, 3400, 3444, 5006, 4500, 400, 4500, 3400, 5400, 0, 0, 0, 4500, 4500, 4400, 555, 4540, 0, 0,
    #           4555, 3000, 4555, 455, 3444, 4333, 2344, 4454, 4555, 3444]
    #demand = [300, 5001, 6022, 6103, 3533, 4046, 3044, 3023, 4064, 9552, 3960, 5077, 4000, 4000, 3430, 300, 3400, 3440,
    #          200, 3400, 9400, 4340, 3400, 300, 3040, 4000, 5000, 6500, 45454, 4443, 3244, 2334, 344, 3223, 2999, 4000, 4000,
    #          4000, 3400, 0, 0, 0, 4500, 400, 4500, 3400, 5400, 3000, 600, 0, 4500, 0, 0, 555, 4540, 500, 800,
    #          4555, 3000, 4555, 455, 3444, 4333, 2344, 4454, 4555, 3444]

    #demand = [0, 0, 0, 6103, 3533, 4046, 3044, 3023, 4064, 9552, 3960, 5077, 4000, 4000, 3430, 300, 3400, 3440,
    #          200, 3400, 9400, 4340, 3400, 300, 3040, 4000, 5000, 6500, 45454, 4443, 3244, 2334, 344, 3223, 2999, 4000, 4000,
    #          4000, 3400, 0, 0, 0, 4500, 400, 4500, 3400, 5400, 3000, 600, 0, 4500, 0, 0, 555, 4540, 500, 800,
    #          4555, 3000, 4555, 455, 3444, 4333, 2344, 4454, 4555, 3444]

    ww = WagnerWhitin()
    output = ww.run(dict_in, demand)
    print(output)
