class Die:
    def __init__(self, lower_left_x, lower_left_y, upper_right_x, upper_right_y):
        self.lower_left_x = lower_left_x
        self.lower_left_y = lower_left_y
        self.upper_right_x = upper_right_x
        self.upper_right_y = upper_right_y
        self.inputs = {}
        self.outputs = {}
        self.instances = {}
        self.placed_sites = set()

    def add_input(self, name, x, y):
        self.inputs[name] = (x, y)

    def add_output(self, name, x, y):
        self.outputs[name] = (x, y)


class FlipFlop:
    def __init__(self, bits, name, width, height, pin_count):
        self.bits = bits
        self.name = name
        self.width = width
        self.height = height
        self.pin_count = pin_count
        self.pins = {}
        self.features = {}

    def add_pin(self, pin_name, x, y):
        self.pins[pin_name] = (x, y)


class Gate:
    def __init__(self, name, width, height, pin_count):
        self.name = name
        self.width = width
        self.height = height
        self.pin_count = pin_count
        self.pins = {}

    def add_pin(self, pin_name, x, y):
        self.pins[pin_name] = (x, y)


class CellLibrary:
    def __init__(self):
        self.flip_flops = {}
        self.gates = {}

    def add_flip_flop(self, flip_flop):
        self.flip_flops[flip_flop.name] = flip_flop

    def add_gate(self, gate):
        self.gates[gate.name] = gate


class Instance:
    def __init__(self, name, cell_type, x, y):
        self.name = name
        self.cell_type = cell_type
        self.x = x
        self.y = y

    def __str__(self):
        return str([self.name, self.cell_type, self.x, self.y])

    def __repr__(self):
        return self.__str__()


class Net:
    def __init__(self, name):
        self.name = name
        self.pins = []

    def add_pin(self, pin):
        self.pins.append(pin)


class Netlist:
    def __init__(self):
        self.nets = {}

    def add_net(self, net):
        self.nets[net.name] = net


class Bin:
    def __init__(self, width, height, max_utilization):
        self.width = width
        self.height = height
        self.max_utilization = max_utilization


class PlacementRow:
    def __init__(self, start_x, start_y, site_width, site_height, total_sites):
        self.start_x = start_x
        self.start_y = start_y
        self.site_width = site_width
        self.site_height = site_height
        self.total_sites = total_sites

    def __str__(self):
        return str([self.start_x, self.start_y, self.site_width, self.site_height, self.total_sites])

    def __repr__(self):
        return str(self)


class TimingInfo:
    def __init__(self, displacement_delay):
        self.displacement_delay = displacement_delay
        self.qpin_delays = {}
        self.timing_slacks = {}

    def add_qpin_delay(self, cell_name, delay):
        self.qpin_delays[cell_name] = delay

    def add_timing_slack(self, instance_name, pin_name, slack):
        self.timing_slacks[(instance_name, pin_name)] = slack


class PowerInfo:
    def __init__(self):
        self.gate_powers = {}

    def add_power(self, cell_name, power):
        self.gate_powers[cell_name] = power
