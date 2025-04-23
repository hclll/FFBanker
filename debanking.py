# Import modules / functions here
from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import find_single_bit_ff
import statistics
import re
import subprocess

# 1) Run checker and get current design's pin timing changes. 
# 2) If slack decreased a large amount, debank MBFFs with pins that had slack decreased. 

def run_checker(input_file,output_file): # Checker must be in same directory as other stuff, inputs should be strings of file names.
    
    result = subprocess.run(["./main",input_file,output_file],capture_output = True);
    success = str(result.stdout); # returns text if checker was successful
    #error = str(result.stderr); # returns text if checker was unsuccessful
    success = success.replace("\\n","\n");
    split_success = success.splitlines();
    
    decreased_slack = {}; # Key = String with new register name, Values = old pins & old and new slacks

    for line in split_success:
        if "timing change on pin" in line:
            split_line = line.split();
            old_pin = split_line[4];
            old_slack = split_line[5];
            new_pin = split_line[6];
            new_slack = split_line[7];

            if float(new_slack) < float(old_slack): # Checks if slack got more negative (worse)
                decreased_slack[new_pin] = {new_slack,old_pin,old_slack};

    return decreased_slack; # Return a dictionary of bad slack pins and associated register names in current design


def debanking_some(die, cell_lib, netlist, decreased_slack): # decreased_slack is a dictionary
    
    new_instances = {}
    ff_count = 0
    single_bit_ff_type = find_single_bit_ff(cell_lib)

    for inst_name, inst in list(die.instances.items()):
        if inst.cell_type in cell_lib.flip_flops:        
            ff = cell_lib.flip_flops[inst.cell_type]

            in_dict = False;
            for key in decreased_slack:
                if str(inst_name) in str(key):
                    in_dict = True; # the current FF was reported as having decreased slack

            # Check that FF is MBFF and that its name is included in the key of the decreased_slack dictionary.
            if ff.bits > 1 and in_dict: 
                print(f"Debanking {inst_name} ({ff.bits}-bit FF) at ({inst.x}, {inst.y})")
                # create bit-indexed instance
                bit_instances = {}
                for bit in range(ff.bits):
                    new_name = inst.original_name; # new name should be old name of original single-bit FF
                    new_inst = Instance(new_name, single_bit_ff_type, inst.x, inst.y + bit * 5)
                    new_inst.original_x = inst.original_x # Don't need to store intermediate mapping info 
                    new_inst.original_y = inst.original_y # The debanked FFs new info should be old instance info
                    new_inst.original_cell_type = inst.original_cell_type  
                    new_inst.original_name = inst.original_name 
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

def generate_output_file(parsed_data):

    # Access cell library
    cell_lib = parsed_data.cell_library;

    # Access instances dictionary
    instances = parsed_data.die.instances;

    # Write instance names and mappings to output.txt file
    with open("output.txt","w") as file:

        # Write number of instances in new design
        file.write("CellInst " + str(len(instances))+"\n");

        # Write instance name, cell name, and coordinates
        for key in instances:
            file.write("Inst " + str(instances[key].name) + " " + str(instances[key].cell_type) + " "
                               + str(instances[key].x)+ " " + str(instances[key].y) + "\n");

        # Map old register pins to new register pins
        for key in instances:

            # Notes:
            # Check if the .pin_mapping dict is empty. If empty, it's a single-bit but NOT debanked register, so same pin names are used.
            # Check if the pin mappings have to go in the order of D, Q, CLK. If so, I have to fix the empty dict case.
            # Add case that checks for banked FFs and maps those. Differentiate multi-bit FF already present in input case vs after banking phase.

            # Check if instance has not had its pins changed (currently, it only changes if it gets debanked in preprocessing)
            if (not instances[key].pin_mapping):
                for pin in cell_lib.flip_flops[instances[key].cell_type].pins:
                    file.write(str(instances[key].original_name) + "/" + str(pin) + " "
                          + "map" + " " + str(instances[key].name) + "/" + str(pin) + "\n");

            # If pin_mapping dict is not empty, this is a single-bit, debanked register. Debanked after preprocessing.
            # New CLK currently is the same as the old CLK, as its pin should not change.
            else:
                for pin in cell_lib.flip_flops[instances[key].cell_type].pins:
                    if (pin == 'D'):
                        for old_pin in instances[key].pin_mapping:
                            if ("D" in str(old_pin)): # check that old_pin is a D-pin
                                file.write(str(instances[key].original_name) + "/" + str(instances[key].pin_mapping[old_pin]) + " "
                                + "map" + " " + str(instances[key].name) + "/" + str(pin) + "\n");
                    elif (pin != 'D' and pin!= "Q"): # should be CLK
                        file.write(str(instances[key].original_name) + "/" + str(pin) + " "
                                + "map" + " " + str(instances[key].name) + "/" + str(pin) + "\n");
                    elif (pin == 'Q'):
                        for old_pin in instances[key].pin_mapping:
                            if ("Q" in str(old_pin)): # check that old_pin is a Q-pin
                                file.write(str(instances[key].original_name) + "/" + str(instances[key].pin_mapping[old_pin]) + " "
                                + "map" + " " + str(instances[key].name) + "/" + str(pin) + "\n");


if __name__ == "__main__":
    parser = Parser("bm/sampleCase")
    parsed_data = parser.parse()

    # Run pre-processing here
    
    # Run banking here

    # Run generate_output_file here 
    generate_output_file(parsed_data);
    input_file = str("sampleCase");
    output_file = str("sampleOutput"); # replace this with output.txt 

    # Run checker using test input and generated output file
    decreased_slack = run_checker(input_file,output_file);
    print("Decreased slack entry found:",decreased_slack);

    # Run debanking_some here using checker results.
    die = parsed_data.die;
    cell_lib = parsed_data.cell_library;
    netlist = parsed_data.netlist;

    debanking_some(die,cell_lib,netlist,decreased_slack); 

    # Placement fixing has to be done before generating final output.txt
    # Run generate_output_file here for final design. Should replace previous version in same directory.


    
    
