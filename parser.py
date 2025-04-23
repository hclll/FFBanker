import re
from module import *

class Parser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.die = None
        self.cell_library = CellLibrary()
        self.instances = []
        self.netlist = Netlist()
        self.bin_constraints = None
        self.placement_rows = []
        self.timing_info = None
        self.power_info = PowerInfo()
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.lambda_ = None

    def parse(self):
        with open(self.file_path, "r") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            tokens = lines[i].split()
            if not tokens:
                i += 1
                continue

            keyword = tokens[0]

            if keyword == "Alpha":
                self.alpha = float(tokens[1])
            elif keyword == "Beta":
                self.beta = float(tokens[1])
            elif keyword == "Gamma":
                self.gamma = float(tokens[1])
            elif keyword == "Lambda":
                self.lambda_ = float(tokens[1])
            elif keyword == "DieSize":
                self.die = Die(*map(int, tokens[1:]))
            elif keyword == "NumInput":
                num_inputs = int(tokens[1])
                for _ in range(num_inputs):
                    i += 1
                    input_tokens = lines[i].split()
                    self.die.add_input(input_tokens[1], int(input_tokens[2]), int(input_tokens[3]))
            elif keyword == "NumOutput":
                num_outputs = int(tokens[1])
                for _ in range(num_outputs):
                    i += 1
                    output_tokens = lines[i].split()
                    self.die.add_output(output_tokens[1], int(output_tokens[2]), int(output_tokens[3]))
            elif keyword == "FlipFlop":
                flip_flop = FlipFlop(int(tokens[1]), tokens[2], int(tokens[3]), int(tokens[4]), int(tokens[5]))
                for _ in range(flip_flop.pin_count):
                    i += 1
                    pin_tokens = lines[i].split()
                    flip_flop.add_pin(pin_tokens[1], int(pin_tokens[2]), int(pin_tokens[3]))
                self.cell_library.add_flip_flop(flip_flop)
            elif keyword == "Gate":
                gate = Gate(tokens[1], int(tokens[2]), int(tokens[3]), int(tokens[4]))
                for _ in range(gate.pin_count):
                    i += 1
                    pin_tokens = lines[i].split()
                    gate.add_pin(pin_tokens[1], int(pin_tokens[2]), int(pin_tokens[3]))
                self.cell_library.add_gate(gate)
            elif keyword == "NumInstances":
                num_instances = int(tokens[1])
                for _ in range(num_instances):
                    i += 1
                    inst_tokens = lines[i].split()
                    self.instances.append(Instance(inst_tokens[1], inst_tokens[2], int(inst_tokens[3]), int(inst_tokens[4])))
            elif keyword == "NumNets":
                num_nets = int(tokens[1])
                for _ in range(num_nets):
                    i += 1
                    net_tokens = lines[i].split()
                    net = Net(net_tokens[1])
                    num_pins = int(net_tokens[2])
                    for _ in range(num_pins):
                        i += 1
                        pin_tokens = lines[i].split()
                        net.add_pin(pin_tokens[1])
                    self.netlist.add_net(net)
            elif keyword == "BinWidth":
                bin_width = int(tokens[1])
            elif keyword == "BinHeight":
                bin_height = int(tokens[1])
            elif keyword == "BinMaxUtil":
                self.bin_constraints = Bin(bin_width, bin_height, float(tokens[1]))
            elif keyword == "PlacementRows":
                self.placement_rows.append(PlacementRow(*map(int, tokens[1:])))
            elif keyword == "DisplacementDelay":
                self.timing_info = TimingInfo(float(tokens[1]))
            elif keyword == "QpinDelay":
                self.timing_info.add_qpin_delay(tokens[1], float(tokens[2]))
            elif keyword == "TimingSlack":
                self.timing_info.add_timing_slack(tokens[1], tokens[2], float(tokens[3]))
            elif keyword == "GatePower":
                self.power_info.add_power(tokens[1], float(tokens[2]))
            else:
                assert False, "Unknown item found"

            i += 1

        self.die.instances = {inst.name: inst for inst in self.instances}

        return self


if __name__ == "__main__":
    parser = Parser("bm/sampleCase")
    parsed_data = parser.parse()

    print("Die Size:", vars(parsed_data.die))
    print("Number of Flip-Flops:", len(parsed_data.cell_library.flip_flops))
    print("Number of Gates:", len(parsed_data.cell_library.gates))
    print("Number of Instances:", len(parsed_data.instances))
    print("Number of Nets:", len(parsed_data.netlist.nets))
    print("Placement Rows:", len(parsed_data.placement_rows))
    print("Timing Info:", vars(parsed_data.timing_info) if parsed_data.timing_info else "None")
    print("Power Info:", vars(parsed_data.power_info))
    print("Alpha:", parsed_data.alpha)
    print("Beta:", parsed_data.beta)
    print("Gamma:", parsed_data.gamma)
    print("Lambda:", parsed_data.lambda_)

    print(parsed_data.instances)
    print(parser.placement_rows)
    print(parser.die.instances)

    print("------------------------------------------------------------------\nNetlist:\n")
    for net in parsed_data.netlist.nets.values():
        print(net.name, net.pins)
