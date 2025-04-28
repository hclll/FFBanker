from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import find_single_bit_ff, debanking_all
from placement import force_directed_placement
from banking import cluster_and_merge_flip_flops, create_pin_mapping, resolve_overlaps, banking_each_clock_net
from output import generate_output_file, generate_default_output_file, check_output
from debanking import run_checker, debanking_some

def main():
    # Preprocessing
    #input_file = str("sampleCase2");
    input_file = str("bm/testcase1_0812.txt")
    output_file = str("output.txt")
    parser = Parser(input_file)  
    parsed_data = parser.parse()
    die = parsed_data.die
    cell_lib = parsed_data.cell_library
    netlist = parsed_data.netlist
    rows = parsed_data.placement_rows
    debanking_all(die,cell_lib,netlist)

    force_directed_placement(die, netlist, rows, cell_lib, iterations=20, damping=0.8)
    print("Placement Completed")
    resolve_overlaps(parsed_data)
    generate_output_file(parsed_data, "Outputs/testcase1_0812_placed.txt")
    
    # Banking
    final_instances, old_to_new_map = banking_each_clock_net(parsed_data)
    create_pin_mapping(die.instances,final_instances,old_to_new_map,cell_lib)
    die.instances = final_instances
    resolve_overlaps(parsed_data)
    generate_output_file(parsed_data, "output_after_banking.txt")

    # Run checker using test input and generated output file.
    input_file = str(input_file)
    output_file = str(output_file)
    decreased_slack = run_checker(input_file,"output_after_banking.txt");

    # Debank more using checker slack results.
    die = parsed_data.die
    cell_lib = parsed_data.cell_library
    netlist = parsed_data.netlist
    debanking_some(die,cell_lib,netlist,decreased_slack,parsed_data)

    resolve_overlaps(parsed_data) # put newly debanked FFs into legal sites

    # Run generate_output_file here for final design.
    generate_output_file(parsed_data, "output_after_debanking.txt")

    input_file = input_file
    output_file = output_file
    check_pass, init_score, final_score = check_output(input_file,"output_after_debanking.txt")
    
    if not check_pass or final_score > init_score:
        print("Generate default output file")
        parser = Parser(input_file)
        parsed_data = parser.parse()
        generate_default_output_file(parsed_data, file_name=output_file)
    
    

if __name__ == "__main__":
    main()
