class UserControl:
    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True
        self.lung_measure = False
        self.set_baseline = True
        self.compute_volume = False
        self.laser = True