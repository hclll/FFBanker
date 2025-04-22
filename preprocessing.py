from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
import statistics
import re

def annotate_ff_features(cell_lib: CellLibrary, timing_info, power_info):
    """
    annnotates flip-flops with features based on their area, delay, and power.
    cell.ff_features = { 'low_power': True, 'low_delay': False, ... }
    """
    bits = [1, 2, 4]
    for bit in bits:
        all_ffs = [ff for ff in cell_lib.flip_flops.values() if ff.bits == bit]

        if not all_ffs:
            print(f"No {bit}-bit FFs found.")
            continue

        # area threshold using mean / median
        areas = [ff.width * ff.height for ff in all_ffs]
        # area_threshold = sum(areas) / len(areas)
        area_threshold = statistics.median(areas)

        # delay/power threshold
        delays = [timing_info.qpin_delays.get(ff.name, float("inf")) for ff in all_ffs]
        powers = [power_info.gate_powers.get(ff.name, float("inf")) for ff in all_ffs]
        # delay_threshold = sum(delays) / len(delays)
        # power_threshold = sum(powers) / len(powers)
        delay_threshold = statistics.median(delays)
        power_threshold = statistics.median(powers)

        for ff in all_ffs:
            area = ff.width * ff.height
            delay = timing_info.qpin_delays.get(ff.name, float("inf"))
            power = power_info.gate_powers.get(ff.name, float("inf"))
            aspect_ratio = ff.height / ff.width if ff.width != 0 else float("inf")

            ff.features = {
                "low_power": power <= power_threshold,
                "low_delay": delay <= delay_threshold,
                "small_area": area <= area_threshold,
                "tall_aspect": aspect_ratio > 2.0,
                "wide_aspect": aspect_ratio < 0.5
            }

def find_single_bit_ff(cell_lib: CellLibrary):
    for ff in cell_lib.flip_flops.values():
        if ff.bits == 1:
            return ff.name
    raise ValueError("No single-bit FF found in cell library!")

def debanking_all(die, cell_lib, netlist):
    new_instances = {}
    ff_count = 0
    single_bit_ff_type = find_single_bit_ff(cell_lib)

    for inst_name, inst in list(die.instances.items()):
        if inst.cell_type in cell_lib.flip_flops:
            ff = cell_lib.flip_flops[inst.cell_type]
            # print(f"Processing {inst_name} ({ff.name}) at ({inst.x}, {inst.y})")
            if ff.bits > 1:
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



if __name__ == "__main__":
    parser = Parser("bm/sampleCase1")
    parsed_data = parser.parse()

    cell_lib = parsed_data.cell_library
    annotate_ff_features(cell_lib, parsed_data.timing_info, parsed_data.power_info)
    for ff in cell_lib.flip_flops.values():
        print(f"FF: {ff.name}, Area: {ff.width * ff.height}, Pins: {ff.pins}, Features: {ff.features}")
    print("///////////////////////////")

    print("instances:", parsed_data.die.instances)
    netlist = parsed_data.netlist
    for net in netlist.nets.values():
        print(f"Net: {net.name}, Pins: {net.pins}")
    print("///////////debanking////////////////")

    debanking_all(parsed_data.die, cell_lib, netlist)
    print("instances:", parsed_data.die.instances)
    for net in netlist.nets.values():
        print(f"Net: {net.name}, Pins: {net.pins}")
    
    for inst_name, inst in parsed_data.die.instances.items():
        print(f"Inst: {inst_name}, Type: {inst.cell_type}, Original Type: {inst.original_cell_type}, Original Name: {inst.original_name}")
        print(inst.pin_mapping)
        # for new_pin, old_pin in inst.pin_mapping.items():
        #     print(f"  Pin mapping: {new_pin} -> {old_pin}")