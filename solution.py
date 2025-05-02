from bus import Bus
from circuit import Circuit
from settings import s
from load import Load
import numpy as np
import pandas as pd

class Solution:

    def __init__(self, name: str, bus: Bus, circuit: Circuit, load: Load):
        self.name = name
        self.bus = bus  # This is a single bus object
        self.circuit = circuit
        self.load = load
        self.delta = dict()  # Will be populated with {bus_name: angle_value}
        self.voltage = dict()  # Will be populated with {bus_name: voltage_value}
        # These should be called after delta and voltage are set
        # so move them out of __init__
        self.P = None
        self.Q = None
        self.x = None
        self.y = None
        self.mismatch = None

    # Initialize with flat start
    def start(self):
        # Use circuit.buses.keys() instead of self.bus.index
    
        angle_deg = [0.00, -4.44, -5.46, -4.70, -4.83, -3.95, 2.15]
        angle_rad = [np.radians(a) for a in angle_deg]

        # Voltage magnitudes (per unit)
        voltage_pu = [1.00000, 0.93692, 0.92049, 0.92980, 0.92672, 0.93968, 0.99999]

        # Bus order (make sure it matches the order in your circuit)
        bus_names = list(self.circuit.buses.keys())

        # Assign values to self.delta and self.voltage
        self.delta = {name: angle for name, angle in zip(bus_names, angle_rad)}
        self.voltage = {name: volt for name, volt in zip(bus_names, voltage_pu)}
        
        # self.delta = {bus_name: 0 for bus_name in self.circuit.buses.keys()}
        # self.voltage = {bus_name: 1 for bus_name in self.circuit.buses.keys()}
        
        # Now calculate the power values
        self.P = self.calc_Px()
        self.Q = self.calc_Qx()
        self.x = self.initialize_x()
        self.y = self.initialize_y()
        self.mismatch = self.calc_mismatch()

    def calc_Px(self):
        # Active power calculation
        # Use circuit.buses.keys() instead of self.bus.index
        Px = {bus_name: 0 for bus_name in self.circuit.buses.keys()}

        for k, bus_k in enumerate(self.circuit.buses.keys()):
            V_k = self.voltage[bus_k]
            delta_k = self.delta[bus_k]
            P_k = 0  # Initialize power for this bus
            
            for j, bus_j in enumerate(self.circuit.buses.keys()):
                V_j = self.voltage[bus_j]
                delta_j = self.delta[bus_j]
                Y_kj = self.circuit.ybus.iloc[k, j] if isinstance(self.circuit.ybus, pd.DataFrame) else self.circuit.ybus[k, j]
                
                # ACCUMULATE power with += instead of reassigning
                P_k += V_k * V_j * abs(Y_kj) * np.cos(delta_k - delta_j - np.angle(Y_kj))

            Px[bus_k] = P_k

        return Px

    def calc_Qx(self):
        # Reactive power calculation
        # Use circuit.buses.keys() instead of self.bus.index
        Qx = {bus_name: 0 for bus_name in self.circuit.buses.keys()}

        for k, bus_k in enumerate(self.circuit.buses.keys()):
            V_k = self.voltage[bus_k]
            delta_k = self.delta[bus_k]
            Q_k = 0  # Initialize reactive power for this bus
            
            for j, bus_j in enumerate(self.circuit.buses.keys()):
                V_j = self.voltage[bus_j]
                delta_j = self.delta[bus_j]
                Y_kj = self.circuit.ybus.iloc[k, j] if isinstance(self.circuit.ybus, pd.DataFrame) else self.circuit.ybus[k, j]
                
                # ACCUMULATE power with += instead of reassigning
                Q_k += V_k * V_j * abs(Y_kj) * np.sin(delta_k - delta_j - np.angle(Y_kj))

            Qx[bus_k] = Q_k

        return Qx

    def initialize_x(self):
        # Create state vector from angles and voltages
        delta_vector = np.array(list(self.delta.values()))
        voltage_vector = np.array(list(self.voltage.values()))
        x = np.concatenate((delta_vector, voltage_vector))
        return x

    def initialize_y(self):
        # This method needs significant revision
        # In the power flow analysis context, we need to create specified power vectors
        # respecting the order and structure needed for the Jacobian matrix
        
        # Get bus type information for determining which values go where
        bus_type_map = {
            "Slack Bus": 1,  # BusType.SLACK
            "PV Bus": 2,     # BusType.PV
            "PQ Bus": 3      # BusType.PQ
        }
        
        # Create mappings for buses with P and Q mismatches
        real_power = []  # P mismatch buses (all except slack)
        reactive_power = []  # Q mismatch buses (only PQ buses)
        
        # for bus_name in self.circuit.buses.keys():
        for bus in self.circuit.buses.values():
            # bus = self.circuit.buses[bus_name]
            if bus.bus_type != 'Slack Bus':
                real_power.append(bus.real_power)

        # for bus_name in self.circuit.buses.keys():
        for bus in self.circuit.buses.values():
            # bus = self.circuit.buses[bus_name]
            if bus.bus_type != 'Slack Bus' and bus.bus_type != 'PV Bus':
                reactive_power.append(bus.reactive_power)
        
        # Combine into a single vector
        y = np.concatenate((np.array(real_power), np.array(reactive_power)))
        y = y / s.base_power
        
        return y

    def calc_mismatch(self):
        # Calculate mismatch between calculated and specified values
        # We'll use a more direct approach that handles the Jacobian structure correctly
        
        # Get bus type information
        bus_type_map = {
            "Slack Bus": 1,  # BusType.SLACK
            "PV Bus": 2,     # BusType.PV
            "PQ Bus": 3      # BusType.PQ
        }
        
        # Create mappings for buses with P and Q equations
        p_buses = []  # P mismatch buses (all except slack)
        q_buses = []  # Q mismatch buses (only PQ buses)
        
        # Identify which buses have P and Q mismatches
        for bus_name, bus in self.circuit.buses.items():
            bus_type = bus_type_map.get(bus.bus_type, bus.bus_type)
            
            if bus_type != 1:  # Not a slack bus
                p_buses.append(bus_name)
            if bus_type == 3:  # PQ bus
                q_buses.append(bus_name)
        
        # Initialize specified power values with zeros
        P_spec = {bus_name: 0 for bus_name in self.circuit.buses.keys()}
        Q_spec = {bus_name: 0 for bus_name in self.circuit.buses.keys()}
        
        # Add load contributions
        if hasattr(self.circuit, 'load') and self.circuit.load:
            # Process all loads in the circuit
            for load_name, load in self.circuit.load.items():
                bus_name = load.bus.name if hasattr(load.bus, 'name') else load.bus
                P_spec[bus_name] -= load.real_power / s.base_power  # Negative for loads
                Q_spec[bus_name] -= load.reactive_power / s.base_power  # Negative for loads
        elif isinstance(self.load, dict):
            for load_name, load in self.load.items():
                P_spec[bus_name] -= load.real_power / s.base_power
                Q_spec[bus_name] -= load.reactive_power / s.base_power
                bus_name = load.bus.name if hasattr(load.bus, 'name') else load.bus
        elif self.load:
            # fallback for a single Load object
            bus_name = self.load.bus.name if hasattr(self.load.bus, 'name') else self.load.bus
            P_spec[bus_name] -= self.load.real_power / s.base_power
            Q_spec[bus_name] -= self.load.reactive_power / s.base_power
        
        # Add generator contributions if available
        for bus_name, bus in self.circuit.buses.items():
            if hasattr(bus, 'p_gen'):
                P_spec[bus_name] += bus.p_gen / s.base_power
            if hasattr(bus, 'q_gen'):
                Q_spec[bus_name] += bus.q_gen / s.base_power
        
        # Calculate power mismatches
        # p_mismatch = [P_spec[bus] - self.P[bus] for bus in p_buses]
        # q_mismatch = [Q_spec[bus] - self.Q[bus] for bus in q_buses]

        #px_value = list(self.calc_Px().values())
        #qx_value = list(self.calc_Qx().values())
        #combined = np.concatenate((px_value, qx_value))
        px = np.array(list(self.calc_Px().values())).reshape(-1, 1)
        qx = np.array(list(self.calc_Qx().values())).reshape(-1, 1)
        combined = np.vstack((px, qx)).flatten()

        y_injected = np.zeros(self.y.size)
        idx = 0

        bus_list = list(self.circuit.buses.values())
        for k in range(len(combined)):
            if k < len(self.circuit.buses):
                bus = bus_list[k]
                if bus.bus_type != 'Slack Bus':
                    y_injected[idx] = combined[k]
                    idx +=1
            else:
                bus = bus_list[k-len(self.circuit.buses)]
                if bus.bus_type != 'Slack Bus' and bus.bus_type != 'PV Bus':
                    y_injected[idx] = combined[k]
                    idx+=1

        # Combine into a single mismatch vector
        mismatch = self.y - y_injected
                    #for i in enumerate(p_buses)]
        #[i[0]]
        
        return mismatch
