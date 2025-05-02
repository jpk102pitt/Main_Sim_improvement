class Conductor:
    def __init__(self, name: str, diam: float, GMR: float, resistance: float, ampacity: float):
        self.name = name
        self.diam = diam
        self.radius = diam / 24 # Need to convert feet to inches (12*2 = 24)
        self.GMR = GMR
        self.resistance = resistance
        self.ampacity = ampacity

if __name__ == '__main__':
    conductor1 = Conductor("Partridge", 0.642, 0.0217, 0.385, 460)

    print(f"name: {conductor1.name}, diam: {conductor1.diam},GMR: , resistance: {conductor1.resistance}, ampacity: {conductor1.ampacity}")
