from bundle import Bundle
from bus import Bus
from conductor import Conductor
from geometry import Geometry
from transformer import Transformer
from transmissionline import TransmissionLine
from load import Load
from generator import Generator
import pandas as pd
from solar_gen import Solar_gen

class Circuit:
    def __init__(self, name: str):
        self.name = name
        self.bundles = dict()
        self.buses = dict()
        self.conductors = dict()
        self.geometries = dict()
        self.transformers = dict()
        self.transmission_lines = dict()
        self.loads = dict()
        self.generators = dict()
        self.solar_gen = dict()
        self.ybus = None

    def add_bundle(self, name, num_conductors, spacing, conductor):
        bundle_obj = Bundle(name, num_conductors, spacing, self.conductors[conductor])
        self.bundles[name] = bundle_obj

    def add_bus(self, name, base_kv):
        bus_obj = Bus(name, base_kv)
        self.buses[name] = bus_obj
        
    def add_conductor(self, name, diam, GMR, resistance, ampacity):
        conductor_obj = Conductor(name, diam, GMR, resistance, ampacity)
        self.conductors[name] = conductor_obj
    
    def add_geometry(self, name, xa, ya, xb, yb, xc, yc):
        geometry_obj = Geometry(name, xa, ya, xb, yb, xc, yc)
        self.geometries[name] = geometry_obj
    
    def add_transformer(self, name, bus1, bus2, power_rating, impedance_percent, x_over_r_ratio):
        transformer_obj = Transformer(name, self.buses[bus1], self.buses[bus2], power_rating, impedance_percent, x_over_r_ratio)
        self.transformers[name] = transformer_obj
    
    def add_transmission_line(self, name, bus1, bus2, bundle, conductor, geometry, length):
        transmission_line_obj = TransmissionLine(name, self.buses[bus1], self.buses[bus2], self.bundles[bundle], self.conductors[conductor], self.geometries[geometry], length)
        self.transmission_lines[name] = transmission_line_obj

    def add_load(self, name, bus, real_power, reactive_power):
        self.loads[name] = Load(name, self.buses[bus], real_power, reactive_power)
        #needs to update the bus real and reactive power

        self.buses[bus].real_power -= real_power
        self.buses[bus].reactive_power -= reactive_power

    def add_generator(self, name, bus, voltage_setpoint, mw_setpoint, x1, x2, x0, zg):
        self.generators[name] = Generator(name, self.buses[bus], voltage_setpoint, mw_setpoint, x1, x2, x0, zg)
        self.buses[bus].real_power += mw_setpoint

    #adding the solar generation
    def add_solar_gen(self, name, bus, y_pv, f_pv, gt, gt_stc, a_p, tc, tc_stc):
        self.solar_gen[name] = Solar_gen(name, self.buses[bus], y_pv, f_pv, gt, gt_stc, a_p, tc, tc_stc)
        self.buses[bus].real_power += self.solar_gen[name].p_pv/1000

    def calc_ybus(self):
        N = len(self.buses)

        # Initialize ybus matrix
        Ybus = pd.DataFrame(index=range(N), columns=range(N)).fillna(0 + 0j).infer_objects(copy=False)

        # Iterate through components
        for component in list(self.transformers.values()) + list(self.transmission_lines.values()):
            Yprim = component.yprim
            bus1_name = component.bus1.name
            bus2_name = component.bus2.name

            if bus1_name in self.buses and bus2_name in self.buses:
                # Get indices for buses in Ybus DataFrame
                bus1_index = list(self.buses.keys()).index(bus1_name)
                bus2_index = list(self.buses.keys()).index(bus2_name)

                # Add self-admittances
                Ybus.iloc[bus1_index, bus1_index] += Yprim[0, 0]
                Ybus.iloc[bus2_index, bus2_index] += Yprim[1, 1]

                # Add mutual admittances
                Ybus.iloc[bus1_index, bus2_index] += Yprim[0, 1]
                Ybus.iloc[bus2_index, bus1_index] += Yprim[1, 0]
            else:
                raise KeyError(f"Buses {bus1_name} or {bus2_name} not found in self.buses.")

        for i in range(N):
            if Ybus.iloc[i, i] == 0:
                raise ValueError(f"Bus {list(self.buses.keys())[i]} has no self-admittance")

        self.ybus = Ybus

    def print_ybus(self):
        if self.ybus is not None:
            with pd.option_context('display.max_rows', 50, 'display.max_columns', 50):
                print("Y-Bus Matrix")
                print(self.ybus)
        else:
            print("Y-Bus not calculated")

if __name__ == '__main__':
    circuit = Circuit("Test Circuit")

    circuit.add_bus("Bus1", 132)
    circuit.add_bus("Bus2", 33)

    circuit.add_conductor("ACSR", 25.76, 0.0111, 0.0891, 780)

    circuit.add_bundle("Bundle1", 2, 0.3, "ACSR")

    circuit.add_geometry("Triangle", 0, 15, -5, 15, 5, 15)

    circuit.add_transformer("T1", "Bus1", "Bus2", 100, 10, 10)

    circuit.add_transmission_line("L1", "Bus1", "Bus2", "Bundle1", "ACSR", "Triangle", 10)

    circuit.calc_ybus()

    circuit.calc_ybus()
    circuit.print_ybus()
