from bus import Bus

class Load:

    def __init__(self, name: str, bus: Bus, real_power: float, reactive_power: float):
        self.name = name
        self.bus = bus
        self.real_power = real_power
        self.reactive_power = reactive_power

if __name__ == '__main__':
    bus1 = Bus("B1", 1.2, "Slack Bus", 1.0, 0.0)

    load1 = Load("L1", bus1, 45, 80)

    print(load1.name, load1.bus, load1.real_power, load1.reactive_power)
