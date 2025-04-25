from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from output import generate_output_file
from banking import resolve_overlaps
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def force_directed_placement(die, netlist, rows, cell_lib, iterations=100, damping=0.8):
    movable = []
    fixed = set()
    positions = {}
    velocities = defaultdict(lambda: np.zeros(2))

    for name, inst in die.instances.items():
        if inst.cell_type in cell_lib.flip_flops:
            movable.append(name)
        else:
            fixed.add(name)
        positions[name] = np.array([inst.x, inst.y], dtype=float)

    for _ in range(iterations):
        forces = defaultdict(lambda: np.zeros(2))

        # force calculation per movable cell 
        for inst_name in movable:
            connected_pins = []  # pins connect to inst_name (exclude itself)
            for net in netlist.nets.values():
                pins = [p[0] for p in net.pins]
                if inst_name in pins:
                    for other_name, _ in net.pins:
                        if other_name != inst_name and other_name != "TOP":
                            connected_pins.append(other_name)

            for other in connected_pins:
                if other in positions:
                    delta = positions[other] - positions[inst_name]
                    distance = np.linalg.norm(delta)
                    if distance > 1e-3:
                        force = delta / distance  # unit vector
                        forces[inst_name] += force  # pull by other

        # update positions 
        for name in movable:
            velocities[name] = (velocities[name] + forces[name]) * damping
            positions[name] = positions[name].astype(float)
            positions[name] += velocities[name]

    for name in movable:
        # snap to site
        best_row = min(rows, key=lambda row: abs(positions[name][1] - row.start_y))
        site_index = round((positions[name][0] - best_row.start_x) / best_row.site_width)
        right_bound = (die.upper_right_x - cell_lib.flip_flops[die.instances[name].cell_type].width - best_row.start_x ) // best_row.site_width
        site_index = max(0, min(site_index, best_row.total_sites - 1, right_bound))

        site_x = best_row.start_x + site_index * best_row.site_width
        site_y = best_row.start_y
        
        # find available site
        offset = 0
        while (site_x + offset * best_row.site_width, site_y) in die.placed_sites and offset < best_row.total_sites - site_index:
            offset += 1
        final_x = site_x + offset * best_row.site_width
        # final_x = site_x
        final_y = site_y

        if final_x > 22995:
            print("right_bound", right_bound)
            print("site_index", site_index)
            print(f"Placing {name} at ({final_x}, {final_y})")

        die.placed_sites.add((final_x, final_y))
        positions[name] = np.array([final_x, final_y])

    print("Final positions:", positions)
    for name in movable:
        die.instances[name].x, die.instances[name].y = positions[name]


if __name__ == "__main__":
    parser = Parser("bm/sampleCase")
    parsed_data = parser.parse()

    cell_lib = parsed_data.cell_library
    netlist = parsed_data.netlist
    die = parsed_data.die
    rows = parsed_data.placement_rows
    force_directed_placement(die, netlist, rows, cell_lib, iterations=500, damping=0.8)
    for inst_name, inst in parsed_data.die.instances.items():
        print(f"Inst: {inst_name}, Position: ({inst.x}, {inst.y})")
    resolve_overlaps(parsed_data)
    generate_output_file(parsed_data)
