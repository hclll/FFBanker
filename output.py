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
    debanking_all(parsed_data.die, cell_lib, netlist)
    
    # Access instances dictionary
    instances = parsed_data.die.instances; 

    # Write instance names and mappings to output.txt file
    with open("output.txt","w") as file:

        # Write number of instances in new design
        file.write("CellInst " + str(len(parsed_data.die.instances))+"\n");

        # Write instance name, cell name, and coordinates
        for key in instances:
            file.write("Inst " + str(instances[key].name) + " " + str(instances[key].cell_type) + " " 
                               + str(instances[key].x)+ " " + str(instances[key].y) + "\n");
        
        # WORK IN PROGRESS: 2 debanked FFs can have the same original cell type, so if I try using a dict to map the old and new pins, 
        #                   both will end up just getting the zero pins as D and Q. I can do a check: if same old cell type, map every 3
        #                   old pins of that old cell type to each debanked FF.   
        
        # Map old register pins to new register pins
        for key in instances:
            
            old_pins = []; 
            new_old_pins_dict = {}; # key = new pin, value = old_pin_key name
            
            # Record the names of the old instance pins
            for old_pin in cell_lib.flip_flops[instances[key].original_cell_type].pins:
                old_pins.append(str(old_pin));  
            print("old_pins: ",old_pins);
            
            # Make a dictionary where new instance pins are keys with corresponding old-pin name values
            for index,new_pin in enumerate(cell_lib.flip_flops[instances[key].cell_type].pins):
                new_old_pins_dict[new_pin] = old_pins[index];
            #print(new_old_pins_dict);
            
            # Map the pins
            for pin in cell_lib.flip_flops[instances[key].cell_type].pins:        
                file.write(str(instances[key].original_name) + "/" + new_old_pins_dict[pin] + " " 
                          + "map" + " " + str(instances[key].name) + "/" + str(pin) + "\n"); 

