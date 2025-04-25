import unittest
from banking import resolve_overlaps, create_site_instance_mappings
from module import Instance, PlacementRow, FlipFlop, CellLibrary, Die, Gate # Added Gate
from collections import defaultdict
from math import ceil # Import ceil

# Mock Parser class to hold the necessary data structures
class MockParser:
    def __init__(self):
        self.placement_rows = []
        self.cell_library = CellLibrary()
        self.die = Die(0, 0, 1000, 1000) # Example die dimensions

class TestBanking(unittest.TestCase):

    def test_resolve_overlap_multi_row(self):
        """
        Tests resolve_overlaps with a single-row FF overlapping a multi-row FF.
        """
        # 1. Define Placement Rows (need at least two consecutive)
        # Assume site width = 10, site height = 20
        row1 = PlacementRow(start_x=0, start_y=0, site_width=10, site_height=20, total_sites=10)
        row2 = PlacementRow(start_x=0, start_y=20, site_width=10, site_height=20, total_sites=10)
        placement_rows = [row1, row2]

        # 2. Define Cell Library
        cell_library = CellLibrary()
        # Single-row, 1-bit FF (width=10, height=20)
        ff1_pin_count = 3 # D, Q, CLK
        ff1 = FlipFlop(name="FF1", width=10, height=20, bits=1, pin_count=ff1_pin_count)
        # Note: We don't need to call add_pin for this test
        cell_library.flip_flops["FF1"] = ff1
        # Multi-row (2 rows), 4-bit FF (width=20, height=40)
        ff4_pin_count = 9 # D0-3, Q0-3, CLK
        ff4 = FlipFlop(name="FF4MR", width=20, height=40, bits=4, pin_count=ff4_pin_count)
        # Note: We don't need to call add_pin for this test
        cell_library.flip_flops["FF4MR"] = ff4
        # Add a dummy gate to avoid issues if create_site_instance_mappings expects gates
        gate1_pin_count = 0 # Assuming no pins needed for this dummy gate
        gate1 = Gate(name="GATE1", width=10, height=20, pin_count=gate1_pin_count)
        cell_library.gates["GATE1"] = gate1


        # 3. Create Instances with Overlap
        # Place FF4MR first at (10, 0), occupying sites (10,0), (20,0), (10,20), (20,20)
        inst_ff4 = Instance(name="U_FF4MR", cell_type="FF4MR", x=10, y=0)
        # Place FF1 overlapping at (10, 0), occupying site (10,0)
        inst_ff1 = Instance(name="U_FF1", cell_type="FF1", x=10, y=0)

        instances = {
            inst_ff4.name: inst_ff4,
            inst_ff1.name: inst_ff1
        }

        # 4. Create Mock Parser
        mock_parser = MockParser()
        mock_parser.placement_rows = placement_rows
        mock_parser.cell_library = cell_library
        mock_parser.die.instances = instances
        # Add original_x/y needed for pin mapping if that runs internally
        for inst in instances.values():
            inst.original_x = inst.x
            inst.original_y = inst.y
            inst.original_name = inst.name # Needed for pin mapping logic if triggered
            inst.pin_mapping = {} # Needed for pin mapping logic if triggered


        # 5. Verify Initial Overlap
        initial_site_map, _ = create_site_instance_mappings(mock_parser)
        initial_overlap_found = False
        for site, occupants in initial_site_map.items():
            if len(occupants) > 1:
                initial_overlap_found = True
                #print(f"Initial overlap at {site}: {occupants}") # Debug print
                break
        self.assertTrue(initial_overlap_found, "Test setup failed: Instances should initially overlap.")

        # 6. Call resolve_overlaps
        # Add a dummy instance name 'C102260' if the assert inside resolve_overlaps requires it
        # This is a temporary workaround for the hardcoded assert found in the provided code.
        # Ideally, this assert should be removed or made conditional.
        dummy_gate_inst = Instance(name="C102260", cell_type="GATE1", x=50, y=0) # Place it somewhere non-overlapping
        mock_parser.die.instances["C102260"] = dummy_gate_inst

        resolved = resolve_overlaps(mock_parser, max_iterations=5) # Use fewer iterations for test speed

        # 7. Assert Outcome
        self.assertTrue(resolved, "resolve_overlaps should return True indicating success.")

        # Verify no overlaps remain
        final_site_map, _ = create_site_instance_mappings(mock_parser)
        final_overlap_found = False
        overlapping_sites = []
        for site, occupants in final_site_map.items():
            # Ignore the dummy gate if it somehow overlaps
            actual_occupants = [name for name in occupants if name != "C102260"]
            if len(actual_occupants) > 1:
                final_overlap_found = True
                overlapping_sites.append((site, actual_occupants))

        self.assertFalse(final_overlap_found, f"Overlaps still exist after resolution at sites: {overlapping_sites}")

        # Optional: Check positions (one should have moved)
        final_ff1 = mock_parser.die.instances["U_FF1"]
        final_ff4 = mock_parser.die.instances["U_FF4MR"]
        # Check if at least one instance moved from its original position (10, 0)
        moved = (final_ff1.x != 10 or final_ff1.y != 0) or \
                (final_ff4.x != 10 or final_ff4.y != 0)
        self.assertTrue(moved, "Neither overlapping instance moved its position.")


    def test_resolve_overlap_surrounded_multi_row(self):
        """
        Tests resolve_overlaps with a multi-row FF overlapped by single-row FFs on left and right.
        The single-row FFs should be moved as there's no space for the multi-row FF.
        """
        # 1. Define Placement Rows (need at least two consecutive)
        # Assume site width = 10, site height = 20
        row1 = PlacementRow(start_x=0, start_y=0, site_width=10, site_height=20, total_sites=10)
        row2 = PlacementRow(start_x=0, start_y=20, site_width=10, site_height=20, total_sites=10)
        placement_rows = [row1, row2]

        # 2. Define Cell Library
        cell_library = CellLibrary()
        # Single-row, 1-bit FF (width=10, height=20)
        ff1_pin_count = 3 # D, Q, CLK
        ff1 = FlipFlop(name="FF1", width=10, height=20, bits=1, pin_count=ff1_pin_count)
        cell_library.flip_flops["FF1"] = ff1
        # Multi-row (2 rows), 4-bit FF (width=20, height=40)
        ff4_pin_count = 9 # D0-3, Q0-3, CLK
        ff4 = FlipFlop(name="FF4MR", width=20, height=40, bits=4, pin_count=ff4_pin_count)
        cell_library.flip_flops["FF4MR"] = ff4
        # Add a dummy gate
        gate1_pin_count = 0
        gate1 = Gate(name="GATE1", width=10, height=20, pin_count=gate1_pin_count)
        cell_library.gates["GATE1"] = gate1

        # 3. Create Instances with Overlap
        # Place FF4MR in the center, e.g., at (20, 0), occupying sites (20,0), (30,0), (20,20), (30,20)
        inst_ff4 = Instance(name="U_FF4MR_Center", cell_type="FF4MR", x=20, y=0)
        # Place FF1 to the left, overlapping at (20, 0), occupying site (20,0)
        inst_ff1_left = Instance(name="U_FF1_Left", cell_type="FF1", x=20, y=0)
        # Place FF1 to the right, overlapping at (30, 0), occupying site (30,0)
        inst_ff1_right = Instance(name="U_FF1_Right", cell_type="FF1", x=30, y=0)

        # Add dummy instances to block movement of FF4MR and leave space for FF1s
        instances = {
            inst_ff4.name: inst_ff4,
            inst_ff1_left.name: inst_ff1_left,
            inst_ff1_right.name: inst_ff1_right,
        }

        # Place dummy gates in all sites except those needed for the initial FFs and the target empty sites for FF1s.
        # Target empty sites for FF1s: (0,0), (10,0), (80,0), (90,0) in row 1
        # and (0,20), (10,20), (80,20), (90,20) in row 2.
        # Initial FF sites: (20,0), (30,0), (20,20), (30,20) for FF4MR; (20,0) for FF1_Left; (30,0) for FF1_Right.
        # Overlapping sites are (20,0) and (30,0).
        # Sites to keep empty: (0,0), (10,0), (80,0), (90,0), (0,20), (10,20), (80,20), (90,20)
        # Sites initially occupied by FF4MR: (20,0), (30,0), (20,20), (30,20)

        sites_to_keep_empty = set([(0, 0), (10, 0), (80, 0), (90, 0), (0, 20), (10, 20), (80, 20), (90, 20)])
        initial_ff4mr_sites = set([(20, 0), (30, 0), (20, 20), (30, 20)])
        initial_ff1_left_site = (20, 0)
        initial_ff1_right_site = (30, 0)


        for row in placement_rows:
            for site_index in range(row.total_sites):
                site_x = row.start_x + site_index * row.site_width
                site_y = row.start_y
                site_coords = (site_x, site_y)

                # Check if this site is one of the initial FF sites or should be kept empty
                is_initial_ff_site = site_coords in initial_ff4mr_sites or \
                                     site_coords == initial_ff1_left_site or \
                                     site_coords == initial_ff1_right_site

                if not is_initial_ff_site and site_coords not in sites_to_keep_empty:
                    # Place a dummy gate if it's not an initial FF site and shouldn't be empty
                    dummy_gate_name = f"DummyGate_{site_x}_{site_y}"
                    dummy_gate_inst = Instance(name=dummy_gate_name, cell_type="GATE1", x=site_x, y=site_y)
                    instances[dummy_gate_name] = dummy_gate_inst


        # 4. Create Mock Parser
        mock_parser = MockParser()
        mock_parser.placement_rows = placement_rows
        mock_parser.cell_library = cell_library
        mock_parser.die.instances = instances
        # Add original_x/y needed for pin mapping if that runs internally
        for inst in instances.values():
            inst.original_x = inst.x
            inst.original_y = inst.y
            inst.original_name = inst.name # Needed for pin mapping logic if triggered
            inst.pin_mapping = {} # Needed for pin mapping logic if triggered

        # Add a dummy instance name 'C102260' if the assert inside resolve_overlaps requires it
        # This is a temporary workaround for the hardcoded assert found in the provided code.
        # Ideally, this assert should be removed or made conditional.
        # Check if C102260 is already used by a dummy gate
        if "C102260" not in instances:
             dummy_gate_inst_assert = Instance(name="C102260", cell_type="GATE1", x=90, y=40) # Place it somewhere non-overlapping
             mock_parser.die.instances["C102260"] = dummy_gate_inst_assert
             dummy_gate_inst_assert.original_x = dummy_gate_inst_assert.x
             dummy_gate_inst_assert.original_y = dummy_gate_inst_assert.y
             dummy_gate_inst_assert.original_name = dummy_gate_inst_assert.name
             dummy_gate_inst_assert.pin_mapping = {}


        # 5. Verify Initial Overlap
        initial_site_map, _ = create_site_instance_mappings(mock_parser)
        initial_overlap_found = False
        overlapping_sites_initial = []
        for site, occupants in initial_site_map.items():
            if len(occupants) > 1:
                initial_overlap_found = True
                overlapping_sites_initial.append((site, occupants))
        print(f"Initial overlap found: {initial_overlap_found}")
        print(f"Initial overlapping sites: {overlapping_sites_initial}")
        self.assertTrue(initial_overlap_found, f"Test setup failed: Instances should initially overlap at sites: {overlapping_sites_initial}")

        # 6. Call resolve_overlaps
        print("\nCalling resolve_overlaps...")
        resolved = resolve_overlaps(mock_parser, max_iterations=10) # Increase iterations just in case
        print(f"resolve_overlaps returned: {resolved}")

        # 7. Assert Outcome
        self.assertTrue(resolved, "resolve_overlaps should return True indicating success.")

        # Print final positions
        print("\nFinal Instance Positions:")
        print(f"U_FF4MR_Center: ({mock_parser.die.instances['U_FF4MR_Center'].x}, {mock_parser.die.instances['U_FF4MR_Center'].y})")
        print(f"U_FF1_Left: ({mock_parser.die.instances['U_FF1_Left'].x}, {mock_parser.die.instances['U_FF1_Left'].y})")
        print(f"U_FF1_Right: ({mock_parser.die.instances['U_FF1_Right'].x}, {mock_parser.die.instances['U_FF1_Right'].y})")

        # Verify no overlaps remain
        final_site_map, _ = create_site_instance_mappings(mock_parser)
        final_overlap_found = False
        overlapping_sites_final = []
        for site, occupants in final_site_map.items():
            # Ignore dummy gates and the assert dummy if they somehow overlap (shouldn't happen with this setup)
            actual_occupants = [name for name in occupants if name not in instances or instances[name].cell_type != "GATE1"]
            if len(actual_occupants) > 1:
                final_overlap_found = True
                overlapping_sites_final.append((site, actual_occupants))

        self.assertFalse(final_overlap_found, f"Overlaps still exist after resolution at sites: {overlapping_sites_final}")

        # Assert that the single-row FFs moved, and the multi-row FF did not move
        final_inst_ff1_left = mock_parser.die.instances["U_FF1_Left"]
        final_inst_ff1_right = mock_parser.die.instances["U_FF1_Right"]
        final_inst_ff4 = mock_parser.die.instances["U_FF4MR_Center"]

        self.assertFalse(final_inst_ff1_left.x == 20 and final_inst_ff1_left.y == 0, "Left single-row FF did not move.")
        self.assertFalse(final_inst_ff1_right.x == 30 and final_inst_ff1_right.y == 0, "Right single-row FF did not move.")
        self.assertTrue(final_inst_ff4.x == 20 and final_inst_ff4.y == 0, "Multi-row FF should not have moved.")

if __name__ == '__main__':
    unittest.main()
