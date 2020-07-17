from MDT693Bops import xyz
import DataIn
from pipeline import pipeline as pip
from read_AI_data import data_reader
from pick_data_set import AI_data_pick

class next_commander(object):

    def __init__(self, pip):

        self.pip = pip
        self.dt = data_reader()
        sefl.pick = AI_data_pick()
        self.xyz = xyz()

    def send_command(self):
        '''sends next command to piezo controller'''

        dec = getattr(self.pip,'next_comm')
        if dec =='x-': self.pip.xv-=0.5
        elif dec =='x+': self.pip.xv+=0.5
        elif dec =='y-':self.pip.yv-=0.5
        elif dec =='y+':self.pip.yv+=0.5
        elif dec =='z+': self.pip.zv-=0.5
        elif dec =='z-': self.pip.zv+=0.5
        if self.pip.xv>150 or self.pip.xv<0 or self.pip.yv<0 or self.pip.yv>150 or self.pip.zv <0 or self.pip.zv >150:
            raise ValueError('Napięcie poza skalą')
        self.xyz.set('x',self.pip.xv)
        self.xyz.set('y',self.pip.yv)
        self.xyz.set('z',self.pip.zv)

    def set_command(self):
        '''sets next command in pipeline'''

        datin = self.pip.xh +self.pip,yh +self.pip.zh
        self.pip.next_comm = self.dt.ret_command(datin)

    def do(self):

        while self.pip.GO:
            self.pick.farmer_do()
            self.set_command()
        
        
        
