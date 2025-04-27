from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import annotate_ff_features, find_single_bit_ff, debanking_all
import statistics
import re
import subprocess

def generate_output_file(parsed_data, file_name="output.txt"): 

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
    with open(file_name,"w") as file:

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
                                file.write(str(instances[key].pin_mapping[new_pin_key][0]) + "/" + str(instances[key].pin_mapping[new_pin_key][1]) + " " + 
                                "map" + " " + str("new_" + instances[key].name) + "/" + str(pin) + "\n");

                    if cell_lib.flip_flops[instances[key].cell_type].bits == 1:
                        file.write(str(instances[key].pin_mapping[new_pin_key][0]) + "/" + str("CLK") + " " + "map" + " " + str("new_" + instances[key].name) +                     "/" + str("CLK") + "\n");
                    else:
                        #print("original_name: ", instances[key].original_name);
                        for original_name in instances[key].original_name:
                            file.write(str(original_name) + "/" + str("CLK") + " " + "map" + " " + str("new_" + instances[key].name) +                     "/" + str("CLK") + "\n");


def check_output(input_file,output_file): # Checker must be in same directory as other stuff, inputs should be strings of file names.
    
    result = subprocess.run(["./main",input_file,output_file],capture_output = True);
    success = str(result.stdout); # returns text if checker was successful
    # print("success: ", success)
    
    success = success.replace("\\n","\n")
    split_success = success.splitlines()
    # print("split_success: ", split_success)

    check_pass = False
    init_score = -1
    final_score = -1
    for line in split_success:
        if "Init score:" in line:
            init_score = re.split(r"[:\s]+", line)[-1]
            init_score = float(init_score)
            print("Init score: ", init_score)
        if "Check pass" in line:
            check_pass = True
            print("Check: ", line)
        if "Final score:" in line:
            final_score = re.split(r"[:\s]+", line)[-1]
            final_score = float(final_score)
            print("Final score: ", final_score)

    print("check_pass: ", check_pass)

    return check_pass, init_score, final_score


def generate_default_output_file(parsed_data, file_name="output.txt"): 
    # Generate a default output file with no changes
    all_instances = parsed_data.die.instances;
    FF_instances = {};
    #Check that instances are FFs.
    for instance in all_instances:                                                                                           
        if all_instances[instance].cell_type in parsed_data.cell_library.flip_flops:
            FF_instances[instance] = all_instances[instance];
    print(FF_instances.items());
    with open(file_name, "w") as file:
        file.write("CellInst " + str(len(FF_instances)) + "\n")
        for inst_name, inst in FF_instances.items():
            file.write(f"Inst new_{inst_name} {inst.cell_type} {inst.x} {inst.y}\n")
        for inst_name, inst in FF_instances.items():
            for pin in parsed_data.cell_library.flip_flops[inst.cell_type].pins:
                file.write(f"{inst_name}/{pin} map new_{inst_name}/{pin}\n")         


if __name__ == "__main__":

    # Parse input file and retrieve data
    parser = Parser("bm/sampleCase1")
    parsed_data = parser.parse()
    cell_lib = parsed_data.cell_library

    # Debank all multi-bit registers in the input file
    netlist = parsed_data.netlist
    debanking_all(parsed_data.die, cell_lib, netlist)
    
    # Generate output file mapping current design to the old design
    generate_output_file(parsed_data)
    generate_default_output_file(parsed_data, file_name="output.txt")
       
    # parser = Parser("bm/sampleCase")
    # generate_default_output_file(parsed_data, file_name="output.txt")
    
    input_file = "bm/sampleCase"
    output_file = "output.txt"
    check_pass, init_score, final_score = check_output(input_file,output_file)
    
    if not check_pass or final_score > init_score:
        print("Generate default output file")
        parser = Parser("bm/sampleCase")
        generate_default_output_file(parsed_data, file_name="output.txt")
    
    


                 

