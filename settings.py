class Settings:

    def __init__(self):
        self.frequency = 60
        self.base_power = 100


s = Settings()
if __name__ == '__main__':

    settings = Settings()
    print(settings.frequency, settings.base_power)

    settings_mod = Settings()
    print(settings_mod.frequency, settings_mod.base_power)
