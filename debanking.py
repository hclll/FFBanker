# Import modules / functions here
from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import find_single_bit_ff, debanking_all
from banking import cluster_and_merge_flip_flops, create_pin_mapping, resolve_overlaps, banking_each_clock_net
from output import generate_output_file
import statistics
import re
import subprocess

# 1) Run checker and get current design's pin timing changes. 
# 2) If slack decreased a large amount, debank MBFFs with pins that had slack decreased. 

def run_checker(input_file,output_file): # Checker must be in same directory as other stuff, inputs should be strings of file names.
    
    result = subprocess.run(["./main",input_file,output_file],capture_output = True);
    success = str(result.stdout); # returns text if checker was successful
    
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

            if float(new_slack) < float(old_slack) and float(new_slack) < 0:# Checks if slack got more negative (worse)
                print("slack got more negative for a pin in the new design");
                decreased_slack[new_pin] = {new_slack,old_pin,old_slack};
                print("Pin in new design with bad slack: ",new_pin);
                print("Pin in old design whose slack got changed: ",old_pin);

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
                
                already_mapped_pins = [];

                for bit in range(ff.bits):
                    new_name = f"{inst.name}_bit{bit}"; 
                    new_inst = Instance(new_name, single_bit_ff_type, inst.x, inst.y + bit * 5)
                    new_inst.original_x = inst.original_x # Don't need to store intermediate mapping info 
                    new_inst.original_y = inst.original_y # The debanked FFs new info should be old instance info
                    new_inst.original_cell_type = inst.original_cell_type  
                    new_inst.original_name = inst.original_name 
                    new_instances[new_name] = new_inst
                    
                    d_mapped = False;
                    q_mapped = False;
                    
                    for MBFF_pin in inst.pin_mapping:
                        if MBFF_pin not in already_mapped_pins:
                            if "D" in str(MBFF_pin):
                                new_inst.pin_mapping["D"] = inst.pin_mapping[MBFF_pin];
                                d_mapped = True;
                            else: # only other type should be Q in pin_mapping
                                new_inst.pin_mapping["Q"] = inst.pin_mapping[MBFF_pin];
                                q_mapped = True;
                            #print("dmap: ",d_mapped);
                            #print("qmap: ",q_mapped);
                            already_mapped_pins.append(MBFF_pin);
                            if d_mapped == True and q_mapped == True:
                                break

                    print("banked FF instance's pin_mapping: ", inst.pin_mapping);
                    print("new instance's pin_mapping: ", new_inst.pin_mapping);                
                del die.instances[inst_name]

                # update netlist
                pattern = re.compile(r"([A-Z]+)(\d+)?")  # e.g., D0, Q1
                for net in netlist.nets.values():
                    new_pins = []
                    for instance_name, pin_name in net.pins:
                        #print("instance_name: ", instance_name);
                        #print("inst.name: ", inst.name);
                        if instance_name == inst.name:
                            match = pattern.fullmatch(pin_name)
                            if match:
                                base, bit_str = match.groups()
                                if bit_str is not None:
                                    bit = int(bit_str)
                                    if bit in bit_instances:
                                        new_pins.append((bit_instances[bit].name, base))
                                        bit_instances[bit].pin_mapping[base] = pin_name
                                        bit_instances[bit].pin_mapping[base] = inst.pin_mapping[(instance_name, pin_name)] # Use the original pin mapping
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
    
    # Preprocessing
    parser = Parser("bm/sampleCase")
    #parser = Parser("bm/testcase1_0812.txt")
    parsed_data = parser.parse()
    die = parsed_data.die;
    cell_lib = parsed_data.cell_library;
    netlist = parsed_data.netlist;
    debanking_all(die,cell_lib,netlist);
    
    # Banking
    final_instances, old_to_new_map = banking_each_clock_net(parsed_data);
    create_pin_mapping(die.instances,final_instances,old_to_new_map,cell_lib);
    die.instances = final_instances
    resolve_overlaps(parsed_data);

    # Generate output file of current design 
    generate_output_file(parsed_data);

    # Run checker using test input and generated output file.
    input_file = str("bm/sampleCase");
    #input_file = str("bm/testcase1_0812.txt");
    output_file = str("output.txt");  
    decreased_slack = run_checker(input_file,output_file);
    #print("Decreased slack dictionary:",decreased_slack);

    # Debank more using checker slack results.
    die = parsed_data.die;
    cell_lib = parsed_data.cell_library;
    netlist = parsed_data.netlist;
    #debanking_some(die,cell_lib,netlist,decreased_slack); 

    # FINAL Placement fixing HERE

    # Run generate_output_file here for final design.
    generate_output_file(parsed_data);
    # Make sure names of new instances do not match any of the old names. Easy fix: just add an increasing value to the end of each name string


    
    
