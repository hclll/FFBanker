from module import Die, CellLibrary, Instance, Netlist, Net, FlipFlop
from parser import Parser
from preprocessing import find_single_bit_ff, debanking_all
from placement import force_directed_placement
from banking import cluster_and_merge_flip_flops, create_pin_mapping, resolve_overlaps, banking_each_clock_net
from output import generate_output_file
from debanking import run_checker, debanking_some

def main():
    # Preprocessing
    parser = Parser("bm/sampleCase")
    # parser = Parser("bm/testcase1_0812.txt")  
    parsed_data = parser.parse()
    die = parsed_data.die
    cell_lib = parsed_data.cell_library
    netlist = parsed_data.netlist
    rows = parsed_data.placement_rows
    debanking_all(die,cell_lib,netlist)

    force_directed_placement(die, netlist, rows, cell_lib, iterations=20, damping=0.8)
    resolve_overlaps(parsed_data)
    generate_output_file(parsed_data, "Outputs/sample_placed.txt")
    # generate_output_file(parsed_data, "Outputs/testcase1_0812_placed.txt")

    # return
    
    # Banking
    final_instances, old_to_new_map = banking_each_clock_net(parsed_data)
    create_pin_mapping(die.instances,final_instances,old_to_new_map,cell_lib)
    die.instances = final_instances
    resolve_overlaps(parsed_data)

    # Generate output file of current design 
    generate_output_file(parsed_data, "Outputs/sample_banked.txt")

    # Run checker using test input and generated output file.
    input_file = str("bm/sampleCase")
    # input_file = str("bm/testcase1_0812.txt")
    output_file = str("output.txt")
    decreased_slack = run_checker(input_file,output_file)
    #print("Decreased slack dictionary:",decreased_slack)

    # Debank more using checker slack results.
    die = parsed_data.die
    cell_lib = parsed_data.cell_library
    netlist = parsed_data.netlist
    debanking_some(die,cell_lib,netlist,decreased_slack)

    # FINAL Placement fixing HERE

    # Run generate_output_file here for final design.
    generate_output_file(parsed_data, "Outputs/sample.txt")

if __name__ == "__main__":
    main()
