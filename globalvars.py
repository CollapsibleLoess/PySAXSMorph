paras_dict = {}
plot_counter = 0
output_folder = ''
output_path = ''
start_time = 0

class AllParams:
    def __init__(self):
        self.density_p = 21.5e-6
        self.maxporod = 10
        self.numporod = 100
        self.minguinier = 1e-5
        self.numguinier = 100

        self.rmin = 1
        self.rmax = 5000
        self.rpts = 1000
        self.kmin = 0.001
        self.kmax = 1
        self.kconst = 150
        self.kpts = 10000
        self.boxsize = 500
        self.boxres = 500
        self.num_waves = 10000

class MatrixPosition:
    def __init__(self, xin, yin, zin):
        self.x = xin
        self.y = yin
        self.z = zin

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z


class MorphVoxel:
    def __init__(self, solid):
        self.group = -1
        self.solid = solid
        self.position = None

    def set_group(self, group):
        self.group = group

    def set_solid(self, solid):
        self.solid = solid

    def set_position(self, position):
        self.position = position

    def get_group(self):
        return self.group

    def get_solid(self):
        return self.solid

    def get_position(self):
        return self.position
