from module import *
from parser import Parser

def calc_hpwl(net, die, cell_library):
    """
    compute net's bounding box HPWL。
    - net: Net object
    - die: Die object (provide instance and IO coordinates)
    - cell_library: CellLibrary (provide pin offset)
    """
    xs, ys = [], []

    for inst_name, pin_name in net.pins:
        if inst_name == "TOP":
            if pin_name in die.inputs:
                x, y = die.inputs[pin_name]
            elif pin_name in die.outputs:
                x, y = die.outputs[pin_name]
            else:
                continue
        else:
            inst = die.instances[inst_name]
            cell_type = inst.cell_type
            pin_offset = cell_library.get_pin_offset(cell_type, pin_name)
            x = inst.x + pin_offset[0]
            y = inst.y + pin_offset[1]

        xs.append(x)
        ys.append(y)

    return (max(xs) - min(xs)) + (max(ys) - min(ys)) if xs and ys else 0






if __name__ == "__main__":
    parser = Parser("bm/sampleCase")
    parsed_data = parser.parse()

    for net in parsed_data.netlist.nets.values():
        hpwl = calc_hpwl(net, parsed_data.die, parsed_data.cell_library)
        print(f"Net {net.name} HPWL: {hpwl}")