# Import modules / functions here
from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
import statistics
import re

# 1) Parse through instances received from banking phase output. This includes instances that were not banked after going through banking phase.
# 2) Calculate slack of each instance and perhaps use a threshold to determine if one is 'acceptable' or not.
# 3) If slack of current MBFF instance is not acceptable, debank it.
     

# Below is the debanking_all code from preprocessing. Debanking phase will look similar, except it will only debank if ff.bits > 1 AND slack is not 'good'.
# must look for inst_name, inst in list of items after banking
# A banked FF that comes out of the banking phase will have old pins and new pins. Old pins will correspond to D, CLK, Q of original cell_lib type
# Its new pins will be D0, D1, etc
# If I debank a banked FF, I need to make its old pins its new pins. new pins = D, Q and old_pins is still D, Q

def debanking_MBFF(die, cell_lib, netlist):
    new_instances = {}
    ff_count = 0
    single_bit_ff_type = find_single_bit_ff(cell_lib)

    for inst_name, inst in list(die.instances.items()):
        if inst.cell_type in cell_lib.flip_flops:
            ff = cell_lib.flip_flops[inst.cell_type]
            # print(f"Processing {inst_name} ({ff.name}) at ({inst.x}, {inst.y})")
            
            good_slack = 0;
            # Check that slack of ff is good or bad. set good_slack == 1 if good, and to 0 if bad.
            # This will be replaced with negative slack calculation code.
            for features in ff.features:
                if (features == "low_delay"):
                    if (ff.features[features] = False):
                        good_slack = 0;
                    else:
                        good_slack = 1;

            if ff.bits > 1 && good_slack == 0:
                print(f"Debanking {inst_name} ({ff.bits}-bit FF) at ({inst.x}, {inst.y})")
                # create bit-indexed instance
                bit_instances = {}
                for bit in range(ff.bits):
                    new_name = f"{inst.name}_bit{bit}"
                    new_inst = Instance(new_name, single_bit_ff_type, inst.x, inst.y + bit * 5)
                    new_inst.original_x = inst.x
                    new_inst.original_y = inst.y
                    new_inst.original_cell_type = inst.cell_type
                    new_inst.original_name = inst.name
                    bit_instances[bit] = new_inst
                    new_instances[new_name] = new_inst
                del die.instances[inst_name]

                # update netlist
                pattern = re.compile(r"([A-Z]+)(\d+)?")  # e.g., D0, Q1
                for net in netlist.nets.values():
                    new_pins = []
                    for instance_name, pin_name in net.pins:
                        if instance_name == inst.name:
                            match = pattern.fullmatch(pin_name)
                            if match:
                                base, bit_str = match.groups()
                                if bit_str is not None:
                                    bit = int(bit_str)
                                    if bit in bit_instances:
                                        new_pins.append((bit_instances[bit].name, base))
                                        bit_instances[bit].pin_mapping[base] = pin_name
                                else:
                                    # duplicate non-bit pins for all bits
                                    for new_inst in bit_instances.values():
                                        new_pins.append((new_inst.name, base))

                        else:
                            new_pins.append((instance_name, pin_name))
                    net.pins = new_pins
            else:
                new_instances[inst_name] = inst
        else:
            new_instances[inst_name] = inst

    die.instances = new_instances


