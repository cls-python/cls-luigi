#!/bin/bash
#!/usr/bin/env  python
# -*- coding: utf-8 -*-

import numpy

class PartPeriod():
    def __init__(self):
        pass

    def run(self, dict_in, demand):
        # Füllen der Variablen aus Start_Dictionary
        dem = demand
        kf = dict_in["fixedCost"]
        kv = dict_in["varCost"]
        pp = len(dem)
        p = 0  # aktuelle Periode
        i = 0
        orders = [0 for i in range(0, pp)]
        orders1 = [0 for i in range(0, pp)]
        # Hilfsvariablen Kosten summieren
        cost = 0
        cost_i = 0
        cost_v = 0

        # testen ob "Null-Perioden am Anfang vorliegen
        while dem[p] == 0:
            orders1[p] = 0
            p = p + 1
            if p == pp:
                p = p - 1
                break
        i = p

        while p < pp:  # solange wie aktuelle Periode kleiner Anzahl PlanungsPerioden
            cost = cost + demand[p] * kv * (p - i)  # Formel Stückausgleichkostenverfahren
            if cost < kf:
                orders[p] += demand[p]
                lot = sum(orders) - sum(orders1)
                cost_v = cost
            else:
                cost_i = cost_v + cost_i
                orders1[i] = lot
                i = p
                p = p - 1
                cost = 0
                cost_v = 0
            p = p + 1
        orders1[i] = lot
        x = orders1.count(0)
        fix = (pp - x) * kf
    #    print(cost_i)
    #    print(fix)
        total_c = cost_i + fix + cost_v
        print("total cost: " + str(total_c))
    #    print(x)
    #    print(sum(orders1))
        output = numpy.array(orders1)
        return output

if __name__ == "__main__":
    # statisches Test dictionary
    dict_in = {
        "planningPeriod": 8,  # Anzahl der Perioden
        "fixedCost": 10000,  # Bestellkosten
        "varCost": 1,  # Lagerhaltungssatz
        "roll": 4
    }

    #demand = [40, 50, 10, 20, 30, 40, 20, 25]
    #demand = [0, 0, 0, 10, 0, 1]
    #demand = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    #demand = [20, 50, 10, 50, 50, 10, 20, 40, 20, 30]
    #demand = [854, 1021, 984, 1030, 1240, 1178, 905, 1005, 958, 905, 966, 965, 1030, 1111, 1285, 1089, 959, 920]
    #demand = [300, 5001, 6022, 6103, 3533, 4046, 3044, 3023, 4064, 9552, 3960, 5077, 4000, 4000, 3430, 300, 3400, 3440,
    #          200, 3400, 9400, 4340, 3400, 300, 3040, 4000, 5000, 6500, 45454, 4443, 3244, 2334, 344, 3223, 2999, 4000, 4000,
    #          4000, 3400, 0, 0, 0, 4500, 400, 4500, 3400, 5400, 3000, 600, 0, 4500, 0, 0, 555, 4540, 500, 800,
    #          4555, 3000, 4555, 455, 3444, 4333, 2344, 4454, 4555, 3444
    demand = [0, 0, 6022, 6103, 3533, 4046, 3044, 3023, 4064, 9552, 3960, 5077, 4000, 4000, 3430, 300, 3400, 3440, 200,
              3400, 9400, 4340, 3400, 300, 3040, 4000, 5000, 6500, 45454, 4443, 3244, 2334, 344, 3223, 2999, 4000, 4000,
              4000, 3400, 3400, 3444, 5006, 4500, 400, 4500, 3400, 5400, 0, 0, 0, 4500, 4500, 4400, 555, 4540, 0, 0,
              4555, 3000, 4555, 455, 3444, 4333, 2344, 4454, 4555, 3444]

    pp = PartPeriod()
    output = pp.run(dict_in, demand)
    print(output)
