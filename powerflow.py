import numpy as np
from jacobian import Jacobian
from solution import Solution

class PowerFlow:
    def __init__(self, circuit):
        self.circuit = circuit
        self.jacobian = Jacobian(circuit)

    def solve_circuit(self, circuit, tol=0.001, max_iter=50):
        buses = list(circuit.buses.values())
        solution = Solution("PowerFlowSolution", buses, circuit, circuit.loads)
        solution.start()

        mismatch_history = []
        converged = False

        for iteration in range(max_iter):
            mismatch = solution.calc_mismatch()
            mismatch_history.append(np.max(np.abs(mismatch)))

            if np.max(np.abs(mismatch)) < tol:
                converged = True
                break

            angles = np.array([solution.delta[bus.name] for bus in buses])
            voltages = np.array([solution.voltage[bus.name] for bus in buses])

            J = self.jacobian.calc_jacobian(buses, circuit.ybus, angles, voltages)
            dx = np.linalg.solve(J, mismatch)

            theta_update = []
            v_update = []
            for bus in buses:
                if bus.bus_type != 'Slack Bus':
                    theta_update.append(bus.name)
            for bus in buses:
                if bus.bus_type == 'PQ Bus':
                    v_update.append(bus.name)

            for i, bus_name in enumerate(theta_update):
                solution.delta[bus_name] += dx[i]

            for j, bus_name in enumerate(v_update):
                solution.voltage[bus_name] += dx[len(theta_update) + j]

            solution.P = solution.calc_Px()
            solution.Q = solution.calc_Qx()

        final_angles = np.array([solution.delta[bus.name] for bus in buses])
        final_voltages = np.array([solution.voltage[bus.name] for bus in buses])

        results = {
            "converged": converged,
            "iterations": iteration + 1,
            "final_mismatch": np.max(np.abs(mismatch)),
            "v_mag": final_voltages,
            "v_ang": final_angles,
            "p_calc": list(solution.P.values()),
            "q_calc": list(solution.Q.values()),
            "mismatch_history": mismatch_history
        }

        return results
