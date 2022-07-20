import pandas as pd
import numpy as np

# code = '''
# * SPICE
# '''

# '''
# * SPICE DP #1

# V  1 0 DC 12
# R1 1 2 10k
# R2 2 3 10k
# R3 3 0 10k

# .op
# .tran 0 10ms 0ms 0.1ms
# .print v(1) v(2) v(3)
# .end

# '''

def toSPICE(circuit, voltage: int, path: str):
    start_network = circuit[circuit['layer'] == 0]
    end_network = circuit[circuit['layer'] == circuit.iloc[-1].layer]

    code = '* SPICE \n\n'

    line = 'V'
    line += f' {end_network.layer.values[0] - 1}'
    line += f' {(prev_point := start_network.layer.values[0])}'
    line += ' DC'
    line += f' {voltage}'
    line += '\n'

    start_idx = start_network.iloc[-1].name

    code += line + '\n'
    line = ''

    node_count = 0

    for idx, row in circuit.iloc[start_idx+1:].iterrows():
        next_point = row.layer

        if row['class'] == 'Line':
            continue

        line += f"{row['name']}"
        line += f" {next_point}"
        line += f" {prev_point}"
        line += f" {int(row['value'])}"
        line += "\n"

        try:
            next_comp = circuit.iloc[idx+1]

            if next_comp.layer != next_point:
                node_count += 1
                prev_point = next_point

            code += line
            line = ""

        except IndexError:
            print("End of circuit")

    code += '\n'
    code += '.op\n'
    code += '.tran 0 10ms 0ms 0.1ms\n'
    code += '.print '
    for n in range(1, node_count+1):
        code += f'v({n}) '
    code += '\n'
    code += '.end \n'

    with open(path, "w") as f:
        f.write(code)

if __name__ == "__main__":
    circuit_1 = pd.read_json("./spice/circuit_detected_1.json")
    circuit_2 = pd.read_json("./spice/circuit_detected_2.json")
    circuit_3 = pd.read_json("./spice/circuit_detected_3.json")

    toSPICE(circuit_1, 12, './spice/test1.cir')
    toSPICE(circuit_2, 12, './spice/test2.cir')
    toSPICE(circuit_3, 12, './spice/test3.cir')
