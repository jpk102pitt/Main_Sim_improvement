import numpy as np
import pandas as pd
from circuit import Circuit
from bus import Bus
from solution import Solution
from settings import s
from load import Load

class BusType:
    SLACK = 1
    PV = 2
    PQ = 3

class Jacobian:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit

    def calc_jacobian(self, buses, ybus, angles, voltages):
        """
        Calculate the full Jacobian matrix for Newton-Raphson power flow
        
        Parameters:
            buses: List of Bus objects
            ybus: Complex admittance matrix (can be numpy array or pandas DataFrame)
            angles: Current voltage angles (radians)
            voltages: Current voltage magnitudes (per unit)
            
        Returns:
            J: The complete Jacobian matrix with proper 2x2 block structure
        """
        # Convert pandas DataFrame to numpy array if needed
        if hasattr(ybus, 'values'):
            ybus = ybus.values
        
        # Map string bus types to numeric types
        bus_type_map = {
            "Slack Bus": BusType.SLACK,
            "PV Bus": BusType.PV,
            "PQ Bus": BusType.PQ
        }
        
        n = len(buses)
        
        # Create mapping between bus indices and their variables
        p_index = []     # Buses that contribute to P equations (all except slack)
        q_index = []     # Buses that contribute to Q equations (only PQ buses)
        theta_index = [] # Buses whose theta is a variable (all except slack)
        v_index = []     # Buses whose V is a variable (only PQ buses)
        
        for i, bus in enumerate(buses):
            # Convert string bus type to numeric type if needed
            bus_type = bus_type_map.get(bus.bus_type, bus.bus_type)
            
            if bus_type != BusType.SLACK:
                p_index.append(i)
                theta_index.append(i)
            if bus_type == BusType.PQ:
                q_index.append(i)
                v_index.append(i)
        
        # Calculate the size of each submatrix
        n_p = len(p_index)      # Number of P equations
        n_q = len(q_index)      # Number of Q equations
        n_theta = len(theta_index)  # Number of theta variables
        n_v = len(v_index)      # Number of V variables
        
        # Calculate the total Jacobian size
        j_size = n_p + n_q
        
        # Verify the dimensions match the expected formula
        n_slack = sum(1 for bus in buses if bus_type_map.get(bus.bus_type, bus.bus_type) == BusType.SLACK)
        n_pq = len(q_index)
        
        expected_size = (n - n_slack) + n_pq
        
        # Verify that our j_size calculation is correct
        assert j_size == expected_size, f"Jacobian size mismatch: {j_size} != {expected_size}"
        
        # Initialize the full Jacobian matrix
        J = np.zeros((j_size, j_size))
        
        # Calculate each submatrix using the correct partial derivatives
        J1 = self._calc_j1(buses, ybus, angles, voltages, p_index, theta_index, bus_type_map)
        J2 = self._calc_j2(buses, ybus, angles, voltages, p_index, v_index, bus_type_map)
        J3 = self._calc_j3(buses, ybus, angles, voltages, q_index, theta_index, bus_type_map)
        J4 = self._calc_j4(buses, ybus, angles, voltages, q_index, v_index, bus_type_map)
        
        # Fill the Jacobian with the submatrices
        # J1 (dP/dδ) - upper left block
        J[:n_p, :n_theta] = J1
        
        # J2 (dP/dV) - upper right block
        J[:n_p, n_theta:] = J2
        
        # J3 (dQ/dδ) - lower left block
        J[n_p:, :n_theta] = J3
        
        # J4 (dQ/dV) - lower right block
        J[n_p:, n_theta:] = J4
        
        return J
    
    # def calc_jacobian_with_solution(self, solution):
    #     """
    #     Calculate the Jacobian matrix using the Solution object
        
    #     Parameters:
    #         solution: Solution object with delta, voltage, and circuit information
            
    #     Returns:
    #         J: The complete Jacobian matrix
    #     """
    #     # Extract data from solution
    #     circuit = solution.circuit
    #     buses = list(circuit.buses.values())
    #     ybus = circuit.ybus
        
    #     # Convert delta and voltage dictionaries to arrays in the correct order
    #     angles = []
    #     voltages = []
    #     for bus_name in circuit.buses.keys():
    #         angles.append(solution.delta[bus_name])
    #         voltages.append(solution.voltage[bus_name])
        
    #     angles = np.array(angles)
    #     voltages = np.array(voltages)
        
    #     # Calculate Jacobian using existing method
    #     return self.calc_jacobian(buses, ybus, angles, voltages)
    
    # def calc_mismatch(self, solution):
    #     """
    #     Calculate power mismatches using Solution object methods
        
    #     Parameters:
    #         solution: Solution object with current state variables
            
    #     Returns:
    #         mismatch: Vector of power mismatches [ΔP; ΔQ]
    #     """
    #     # Get bus and circuit information
    #     circuit = solution.circuit
    #     buses = circuit.buses
        
    #     # Calculate the actual power injections using Solution methods
    #     P_calc = solution.calc_Px()
    #     Q_calc = solution.calc_Qx()
        
    #     # Get the specified powers for each bus
    #     P_spec = {}
    #     Q_spec = {}
        
    #     # Map string bus types to numeric types
    #     bus_type_map = {
    #         "Slack Bus": BusType.SLACK,
    #         "PV Bus": BusType.PV,
    #         "PQ Bus": BusType.PQ
    #     }
        
    #     # Determine which buses participate in mismatch calculations
    #     p_buses = []  # Buses that have P mismatch (all except slack)
    #     q_buses = []  # Buses that have Q mismatch (only PQ buses)
        
    #     for bus_name, bus in buses.items():
    #         bus_type = bus_type_map.get(bus.bus_type, bus.bus_type)
            
    #         # Get specified values from the bus or load information
    #         if hasattr(bus, 'p_specified'):
    #             P_spec[bus_name] = bus.p_specified
    #         else:
    #             # If bus doesn't have specified power, use 0 as default
    #             # (this should be updated to properly get specified power from loads)
    #             P_spec[bus_name] = 0
                
    #         if hasattr(bus, 'q_specified'):
    #             Q_spec[bus_name] = bus.q_specified
    #         else:
    #             # If bus doesn't have specified power, use 0 as default
    #             # (this should be updated to properly get specified power from loads)
    #             Q_spec[bus_name] = 0
            
    #         # Determine which buses participate in mismatch calculations
    #         if bus_type != BusType.SLACK:
    #             p_buses.append(bus_name)
    #         if bus_type == BusType.PQ:
    #             q_buses.append(bus_name)
        
    #     # Apply load information if available
    #     if hasattr(circuit, 'load') and circuit.load:
    #         for load_name, load in circuit.load.items():
    #             bus_name = load.bus.name if hasattr(load.bus, 'name') else load.bus
    #             # Add load to specified power (convert to per unit)
    #             P_spec[bus_name] += load.real_power / s.base_power
    #             Q_spec[bus_name] += load.reactive_power / s.base_power
    #     elif hasattr(solution, 'load') and solution.load:
    #         bus_name = solution.load.bus.name if hasattr(solution.load.bus, 'name') else solution.load.bus
    #         # Add load to specified power (convert to per unit)
    #         P_spec[bus_name] += solution.load.real_power / s.base_power
    #         Q_spec[bus_name] += solution.load.reactive_power / s.base_power
        
    #     # Calculate mismatches (P_specified - P_calculated, Q_specified - Q_calculated)
    #     P_mismatch = []
    #     Q_mismatch = []
        
    #     for bus_name in p_buses:
    #         P_mismatch.append(P_spec[bus_name] - P_calc[bus_name])
            
    #     for bus_name in q_buses:
    #         Q_mismatch.append(Q_spec[bus_name] - Q_calc[bus_name])
        
    #     # Combine into a single mismatch vector [ΔP; ΔQ]
    #     mismatch = np.concatenate((np.array(P_mismatch), np.array(Q_mismatch)))
        
    #     return mismatch
    
    # def newton_raphson_iteration(self, solution, max_iterations=10, tolerance=1e-6):
    #     """
    #     Perform Newton-Raphson power flow iterations
        
    #     Parameters:
    #         solution: Solution object with initial state
    #         max_iterations: Maximum number of iterations
    #         tolerance: Convergence tolerance for power mismatches
            
    #     Returns:
    #         solution: Updated solution object after convergence
    #         iterations: Number of iterations performed
    #         converged: Boolean indicating whether the solution converged
    #     """
    #     converged = False
    #     iterations = 0
        
    #     while not converged and iterations < max_iterations:
    #         # Calculate mismatch
    #         mismatch = self.calc_mismatch(solution)
            
    #         # Check for convergence
    #         if np.max(np.abs(mismatch)) < tolerance:
    #             converged = True
    #             break
                
    #         # Calculate Jacobian
    #         J = self.calc_jacobian_with_solution(solution)
            
    #         # Solve for state variable updates
    #         delta_x = np.linalg.solve(J, mismatch)
            
    #         # Map string bus types to numeric types
    #         bus_type_map = {
    #             "Slack Bus": BusType.SLACK,
    #             "PV Bus": BusType.PV,
    #             "PQ Bus": BusType.PQ
    #         }
            
    #         # Update state variables
    #         # Determine indices for state variables
    #         buses = list(solution.circuit.buses.values())
    #         n = len(buses)
            
    #         # Create mapping between bus indices and their variables
    #         p_index = []     # Buses that contribute to P equations (all except slack)
    #         q_index = []     # Buses that contribute to Q equations (only PQ buses)
    #         theta_index = [] # Buses whose theta is a variable (all except slack)
    #         v_index = []     # Buses whose V is a variable (only PQ buses)
            
    #         # Map each bus type to their indices
    #         for i, (bus_name, bus) in enumerate(solution.circuit.buses.items()):
    #             bus_type = bus_type_map.get(bus.bus_type, bus.bus_type)
                
    #             if bus_type != BusType.SLACK:
    #                 p_index.append(i)
    #                 theta_index.append(i)
    #             if bus_type == BusType.PQ:
    #                 q_index.append(i)
    #                 v_index.append(i)
            
    #         # Extract updates for delta and voltage
    #         n_theta = len(theta_index)
    #         delta_theta = delta_x[:n_theta]
    #         delta_v = delta_x[n_theta:]
            
    #         # Apply updates to solution object
    #         for i, theta_idx in enumerate(theta_index):
    #             bus_name = list(solution.circuit.buses.keys())[theta_idx]
    #             solution.delta[bus_name] += delta_theta[i]
                
    #         for i, v_idx in enumerate(v_index):
    #             bus_name = list(solution.circuit.buses.keys())[v_idx]
    #             solution.voltage[bus_name] += delta_v[i]
            
    #         # Recalculate powers with updated state
    #         solution.P = solution.calc_Px()
    #         solution.Q = solution.calc_Qx()
            
    #         iterations += 1
        
    #     # Update the final mismatch
    #     solution.mismatch = self.calc_mismatch(solution)
        
    #     return solution, iterations, converged
    
    def _calc_j1(self, buses, ybus, angles, voltages, p_index, theta_index, bus_type_map):
        """
        Calculate J1 submatrix (dP/dδ)
        
        This forms the upper left part of the Jacobian matrix.
        """
        n_p = len(p_index)
        n_theta = len(theta_index)
        j1 = np.zeros((n_p, n_theta))
        all_buses = [bus for bus in self.circuit.buses.values() if bus.bus_type != 'Slack Bus']
        bus_names = list(self.circuit.buses.values())
        for i, bus_i in enumerate(all_buses):
            k = bus_names.index(bus_i)
            for j, bus_j in enumerate(all_buses):
                l = bus_names.index(bus_j)
                if k == l:
                    # Diagonal elements (n = k case)
                    sum_term = 0
                    for n in range(len(buses)):
                        if n != k:  # n ≠ k
                            # y_in = ybus[i, n]
                            y_in_abs = abs(ybus[k, n])
                            theta_in = np.angle(ybus[k, n])
                            sum_term += y_in_abs * voltages[n] * np.sin(angles[k] - angles[n] - theta_in)
                    
                    j1[i, j] = -voltages[k] * sum_term
    
                else:
                    # Off-diagonal elements
                    y_ij = ybus[k, l]
                    y_ij_abs = abs(y_ij)
                    theta_ij = np.angle(y_ij)
                    # j1[i_idx, j_idx] = voltages[i] * voltages[j] * (g_ij * np.sin(theta_ij) - b_ij * np.cos(theta_ij))
                    j1[i, j] = voltages[k] * voltages[l] * y_ij_abs * np.sin(angles[k] - angles[l] - theta_ij)
        
        return j1
    
    def _calc_j2(self, buses, ybus, angles, voltages, p_index, v_index, bus_type_map):
        """
        Calculate J2 submatrix (dP/dV)
        
        This forms the upper right part of the Jacobian matrix.
        """
        n_p = len(p_index)
        n_v = len(v_index)
        j2 = np.zeros((n_p, n_v))
        all_buses = [bus for bus in self.circuit.buses.values() if bus.bus_type != 'Slack Bus']
        pq_buses = [bus for bus in self.circuit.buses.values() if bus.bus_type != 'Slack Bus' and bus.bus_type != "PV Bus"]
        bus_names = list(self.circuit.buses.values())
        for i, bus_i in enumerate(all_buses):
            k = bus_names.index(bus_i)
            for j, bus_j in enumerate(pq_buses):
                l = bus_names.index(bus_j)
                if k == l:
                    # Diagonal elements (n = k case)
                    sum_term = 0

                    y_ii = ybus[k,l]
                    y_ii_abs = abs(y_ii)
                    theta_ii = np.angle(y_ii)
                    first_term = voltages[k] * y_ii_abs * np.cos(theta_ii)

                    for n in range(len(buses)):
                        y_in = ybus[k, n]
                        y_in_abs = abs(y_in)
                        theta_in = np.angle(y_in)
                        sum_term += y_in_abs * voltages[n] * np.cos(angles[k] - angles[n] - theta_in)
                    
                    j2[i, j] = first_term + sum_term
    
                else:
                    # Off-diagonal elements
                    y_ij = ybus[k, l]
                    y_ij_abs = abs(y_ij)
                    theta_ij = np.angle(y_ij)
                    # j2[i_idx, j_idx] = voltages[i] * voltages[j] * (g_ij * np.sin(theta_ij) - b_ij * np.cos(theta_ij))
                    j2[i, j] = voltages[k] * y_ij_abs * np.cos(angles[k] - angles[l] - theta_ij)
        
        return j2
    
    def _calc_j3(self, buses, ybus, angles, voltages, q_index, theta_index, bus_type_map):
        """
        Calculate J3 submatrix (dQ/dδ)
        
        This forms the lower left part of the Jacobian matrix.
        """
        n_p = len(q_index)
        n_theta = len(theta_index)
        j3 = np.zeros((n_p, n_theta))
        all_buses = [bus for bus in self.circuit.buses.values() if bus.bus_type != 'Slack Bus']
        pq_buses = [bus for bus in self.circuit.buses.values() if bus.bus_type != 'Slack Bus' and bus.bus_type != "PV Bus"]
        bus_names = list(self.circuit.buses.values())
        for i, bus_i in enumerate(pq_buses):
            k = bus_names.index(bus_i)
            for j, bus_j in enumerate(all_buses):
                l = bus_names.index(bus_j)
                if k == l:
                    # Diagonal elements (n = k case)
                    sum_term = 0

                    for n in range(len(buses)):
                        if n != k:  # n ≠ k
                            y_in = ybus[k, n]
                            y_in_abs = abs(y_in)
                            theta_in = np.angle(y_in)
                            sum_term += y_in_abs * voltages[n] * np.cos(angles[k] - angles[n] - theta_in)
                    
                    j3[i, j] = voltages[k] * sum_term
    
                else:
                    # Off-diagonal elements
                    y_ij = ybus[k, l]
                    y_ij_abs = abs(y_ij)
                    theta_ij = np.angle(y_ij)
                    # j3[i_idx, j_idx] = voltages[i] * voltages[j] * (g_ij * np.sin(theta_ij) - b_ij * np.cos(theta_ij))
                    j3[i, j] = -voltages[k] * voltages[l] * y_ij_abs * np.cos(angles[k] - angles[l] - theta_ij)
        
        return j3
    
    def _calc_j4(self, buses, ybus, angles, voltages, q_index, v_index, bus_type_map):
        """
        Calculate J4 submatrix (dQ/dV)
        
        This forms the lower right part of the Jacobian matrix.
        """
        n_p = len(q_index)
        n_v = len(v_index)
        j4 = np.zeros((n_p, n_v))
        pq_buses = [bus for bus in self.circuit.buses.values() if bus.bus_type != 'Slack Bus' and bus.bus_type != "PV Bus"]
        bus_names = list(self.circuit.buses.values())
        for i, bus_i in enumerate(pq_buses):
            k = bus_names.index(bus_i)
            for j, bus_j in enumerate(pq_buses):
                l = bus_names.index(bus_j)
                if k == l:
                    # Diagonal elements (n = k case)
                    sum_term = 0

                    y_ii = ybus[k,l]
                    y_ii_abs = abs(y_ii)
                    theta_ii = np.angle(y_ii)
                    first_term = -voltages[k] * y_ii_abs * np.sin(theta_ii)

                    for n in range(len(buses)):
                        y_in = ybus[k, n]
                        y_in_abs = abs(y_in)
                        theta_in = np.angle(y_in)
                        sum_term += y_in_abs * voltages[n] * np.sin(angles[k] - angles[n] - theta_in)
                    
                    j4[i, j] = first_term + sum_term
    
                else:
                    # Off-diagonal elements
                    y_ij = ybus[k, l]
                    y_ij_abs = abs(y_ij)
                    theta_ij = np.angle(y_ij)
                    # j4[i_idx, j_idx] = voltages[i] * voltages[j] * (g_ij * np.sin(theta_ij) - b_ij * np.cos(theta_ij))
                    j4[i, j] = voltages[k] * y_ij_abs * np.sin(angles[k] - angles[l] - theta_ij)
        
        return j4

# Example usage with Solution class:
if __name__ == '__main__':
    # Create a test circuit
    circuit = Circuit("Test Circuit")
        
    # Add buses with different types
    bus1 = Bus("Bus1", 132)
    bus1.bus_type = 'Slack Bus'
    bus1.vpu = 1.0  # Set the slack bus voltage (per unit)
    bus1.delta = 0.0  # Set the slack bus angle (radians)

    bus2 = Bus("Bus2", 132)
    bus2.bus_type = 'PV Bus'
    bus2.vpu = 1.02  # PV buses have specified voltage
    bus2.p_gen = 50  # MW, generator output
    bus2.delta = 0.0  # Initial angle

    bus3 = Bus("Bus3", 33)
    bus3.bus_type = 'PQ Bus'
    bus3.vpu = 1.0  # Initial voltage estimate
    bus3.delta = 0.0  # Initial angle

    # Add buses to the circuit
    circuit.buses = {"Bus1": bus1, "Bus2": bus2, "Bus3": bus3}

    # Create a sample Ybus matrix (admittance matrix)
    circuit.ybus = pd.DataFrame([
        [complex(1.5, -4.0), complex(-0.5, 1.0), complex(-1.0, 3.0)],
        [complex(-0.5, 1.0), complex(1.0, -3.0), complex(-0.5, 2.0)],
        [complex(-1.0, 3.0), complex(-0.5, 2.0), complex(1.5, -5.0)]
    ])

    # Create a load at Bus3
    load = Load("Load1", "Bus3", 50, 30)  # 50 MW, 30 MVAr

    # Create Jacobian instance
    jacobian = Jacobian()

    print(jacobian.calc_jacobian())
