from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import annotate_ff_features, find_single_bit_ff, debanking_all
import statistics
import re

def generate_output_file(parsed_data): 

    # Access cell library
    cell_lib = parsed_data.cell_library;

    # Access instances dictionary
    all_instances = parsed_data.die.instances;
    instances = {};

    # Check that instances are FFs.
    for instance in all_instances:
        #all_instances[instance].name = str("new_" + all_instances[instance].name);
        if all_instances[instance].cell_type in cell_lib.flip_flops:
            instances[instance] = all_instances[instance];

    # Write instance names and mappings to output.txt file
    with open("output.txt","w") as file:

        # Write number of instances in new design
        file.write("CellInst " + str(len(instances))+"\n");

        # Write instance name, cell name, and coordinates
        for key in instances:
            file.write("Inst " + str("new_" + instances[key].name) + " " + str(instances[key].cell_type) + " "
                               + str(instances[key].x)+ " " + str(instances[key].y) + "\n");

        # Map old register pins to new register pins
        for key in instances:

            # Check if instance has not had its pins changed (only possible if it is a single-bit FF that did not get banked)
            if (not instances[key].pin_mapping): 
                for pin in cell_lib.flip_flops[instances[key].cell_type].pins:
                    file.write(str(instances[key].name) + "/" + str(pin) + " "
                          + "map" + " " + str("new_" + instances[key].name) + "/" + str(pin) + "\n");

            # If pin_mapping dict is not empty, this FF got debanked/banked at some point.
            else:
                
                # Check if this is a single-bit register (may not be necessary, so i set it to > 0)
                if cell_lib.flip_flops[instances[key].cell_type].bits > 0:
                    # Go through each pin and see what each maps to
                    for pin in cell_lib.flip_flops[instances[key].cell_type].pins:
                        for new_pin_key in instances[key].pin_mapping:
                            if pin == new_pin_key:
                                file.write(str(instances[key].pin_mapping[new_pin_key][0]) + "/" + str(instances[key].pin_mapping[new_pin_key][1]) + " " + "map" + " " 
                                           + str("new_" + instances[key].name) + "/" + str(pin) + "\n");
                    file.write(str(instances[key].pin_mapping[new_pin_key][0]) + "/" + str("CLK") + " " + "map" + " " + str(instances[key].name) + "/" + str("CLK") + "\n");
                

if __name__ == "__main__":

    # Parse input file and retrieve data
    parser = Parser("bm/sampleCase1")
    parsed_data = parser.parse()
    cell_lib = parsed_data.cell_library

    # Debank all multi-bit registers in the input file
    netlist = parsed_data.netlist
    debanking_all(parsed_data.die, cell_lib, netlist)
    
    # Generate output file mapping current design to the old design
    generate_output_file(parsed_data);
       
    


                 

