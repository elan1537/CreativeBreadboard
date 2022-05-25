import numpy as np

def calcCurrentAndVoltage(V, circuit):
    R_value = []
    for level in circuit:			# 합성저항 구하기
        R_value1 = []
        for n in range(1):
            for element in level:
                R_value1.append(1/element["value"])			
        R_value.append(R_value1)			

    result0 = 1/sum(R_value[0])
    result1 = 1/sum(R_value[1])
    result2 = 1/sum(R_value[2])

    r = result0 + result1 + result2

    I = V/r

    i = []
    for level in circuit: 			# 각 저항에 걸리는 전압
        R_voltage1 = []
        for n in range(1):
            i0 = []
            for element in level:
                R_voltage1.append(element["value"])
                i0 = [I * R_vol for R_vol in R_voltage1]
            i.append(i0)
    
    return r, I, i

if __name__ == "__main__":
    V = 5
    circuit = [
        [	      
            {"name": "R10", "value": 3},
            {"name": "R11", "value": 3},
            {"name": "R11", "value": 3},
        ],
        [     
            {"name": "R21", "value": 2},
            {"name": "R21", "value": 2},
        ], 
        [
            {"name": "R30", "value": 6},
            {"name": "R31", "value": 6},
            {"name": "R31", "value": 6},
        ],
    ]
    R_TH, I, NODE_VOL = calcCurrentAndVoltage(V, circuit)

    print(R_TH, I, NODE_VOL)