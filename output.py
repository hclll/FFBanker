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
        if all_instances[instance].cell_type in cell_lib.flip_flops:
            instances[instance] = all_instances[instance];

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

    # Parse input file and retrieve data
    parser = Parser("bm/sampleCase1")
    parsed_data = parser.parse()
    
    # Calculate median FF power, area, timing and label FFs based on medians
    cell_lib = parsed_data.cell_library
    annotate_ff_features(cell_lib, parsed_data.timing_info, parsed_data.power_info)

    # Debank all multi-bit registers in the input file
    netlist = parsed_data.netlist
    debanking_all(parsed_data.die, cell_lib, netlist)
    
    # Generate output file mapping current design to the old design
    generate_output_file(parsed_data);
        
    


                 

