import warnings

class Bus:
    bus_count = 0

    def __init__(self, name: str, base_kv: float):
        self.name = name
        self.base_kv = base_kv
        self.index = Bus.bus_count
        Bus.bus_count += 1
        self.bus_type = 'Slack Bus'
        self.vpu = 1.0
        self.delta = 0.0
        #need a way to store real and reactive power

        self.real_power = 0.0
        self.reactive_power = 0.0
        
if __name__ == '__main__':
    bus1 = Bus("Bus 1", 20, "PV Bus")
    bus2 = Bus("Bus 2", 230, "PQ Bus")

    print(f"Bus 1: {bus1.name}, {bus1.base_kv}, {bus1.index}, {bus1.vpu}, {bus1.delta}")
    print(f"Bus 2: {bus2.name}, {bus2.base_kv}, {bus2.index}, {bus2.vpu}, {bus2.delta}")
    print(f"Bus count: {Bus.bus_count}")
