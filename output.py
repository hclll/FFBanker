from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import annotate_ff_features, find_single_bit_ff, debanking_all
import statistics
import re

if __name__ == "__main__":

    # Parse input file and retrieve data
    parser = Parser("bm/sampleCase1")
    parsed_data = parser.parse()
    
    # Calculate median FF power, area, timing and label FFs based on medians
    cell_lib = parsed_data.cell_library
    annotate_ff_features(cell_lib, parsed_data.timing_info, parsed_data.power_info)

    # Debank all multi-bit registers in the input file
    netlist = parsed_data.netlist
    #debanking_all(parsed_data.die, cell_lib, netlist)
    
    # Access instances dictionary
    instances = parsed_data.die.instances; 
    
    # Print netlist (I'm trying to figure out where the pin mappings are stored.
    #print(netlist.nets);

    # Write instance names and mappings to output.txt file
    with open("output.txt","w") as file:

        # Write number of instances in new design
        file.write("CellInst " + str(len(parsed_data.die.instances))+"\n");

        # Write instance name, cell name, and coordinates
        for key in instances:
            file.write("Inst " + str(instances[key].name) + " " + str(instances[key].cell_type) + " " 
                               + str(instances[key].x)+ " " + str(instances[key].y) + "\n");

        # Map old register pins to new register pins
        for key in instances:
            for pin in cell_lib.flip_flops[instances[key].cell_type].pins:
                file.write(str(instances[key].original_name) + "/" + str(pin) + " " 
                          + "map" + " " + str(instances[key].name) + "/" + str(pin) + "\n"); 

