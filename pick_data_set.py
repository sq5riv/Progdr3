import DataIn
from ufunctions import UF_sum as uf
from pipeline import pipeline as pipeline
from MDT693Bops import xyz
from statistics import mean
from statistics import median

'''this code takes dataset for learning AI'''

class AI_data_pick(object):
    '''takes data and do work'''

    def __init__(self, pipeline):
        '''Initialise all objects'''

        self.pip = pipeline
        self.fin = DataIn.filmin(self.pip)
        ob_list = []
        #ob_list.append(DataIn.blur_frame(self.pip))
        ob_list.append(DataIn.line_cather_y(self.pip))
        ob_list.append(DataIn.line_catcher_x(self.pip))
        ob_list.append(DataIn.smoother_AI(self.pip,'h_line'))
        ob_list.append(DataIn.smoother_AI(self.pip,'AI_c_line'))
        ob_list.append(DataIn.smoother_AI(self.pip,'AI_r_line'))
        ob_list.append(DataIn.cut_AI(self.pip,'h_line'))
        ob_list.append(DataIn.cut_AI(self.pip,'AI_c_line'))
        ob_list.append(DataIn.cut_AI(self.pip,'AI_r_line'))
        self.ob = ob_list
        self.ufunc = uf(self.pip)
        self.cont = xyz()
        DataIn.play_film(self.pip).start()
        self.f = open('AI_dat.txt','a')
        self.f_run = True

    def calc(self):
        '''calculates all diferences and have info about max'''

        if self.f_run == True:
            self.ox ,self.oy, self.oz = self.val
            self.ouf = self.uf
            self.f_run = False
        else:
            self.x, self.y, self.z = self.val
            self.dx = self.x - self.ox
            self.dy = self.y - self.oy
            self.dz = self.z - self.oz
            self.duf = self.uf - self.ouf
            self.ox,self.oy,self.oz = self.val
            self.ouf = self.uf
            
        
    def conv(self, d):
        '''converts float to string'''

        d = str(round(d,2))
        return d
    
    def end(self):
        '''finish work'''

        self.f.close()
        self.cont.close()
        
    def farmer(self):
        '''do all work'''

        j = 0
        tmp_u = [0]*40
        while True:
            if self.pip.GO == False:
                self.end()
                break
            else:
                self.fin.stp()
                for i in self.ob:
                    i.run()
                tmp_u.append(float(self.ufunc.do()))
                self.uf = median(tmp_u)
                tmp_u = tmp_u[1:]
                if tmp_u[0] == 0: continue
                self.val = self.cont.state()
                self.calc()
                self.save()

    def farmer_do(self):

        if self.pip.GO ==False:
            self.end()
        else:
            self.fin.stp()
            for i in self.ob:
                i.run()
            tmp_u.append(float(self.ufunc.do()))
            self.uf=median(tmp_u)
            tmp_u = tmp_u[1:]
            self.val = self.cont.dtate()
            self.calc()
            self.savepip()
                         
    def save(self):
        '''saves data for file'''

        val = self.conv(self.x)+' '+ self.conv(self.dx)+' '+self.conv(self.y)+' '+self.conv(self.dy)+' '+self.conv(self.z)+' '+self.conv(self.dz)+' '+self.conv(self.uf)+' '+self.conv(self.duf)+'\n'
        if self.duf !=0: self.f.write(val)
              
    def savepip(self):
        '''saves all data for pip'''

        self.s2('xh', self.dx)
        self.s2('yh', self.dy)
        self.s2('zh', self.dz)
        self.s2('ufh', self.duf)
        
    def s2(self, field, dat):
        '''save data for '''

        b = getattr(sefl.pip,field)
        b.append(dat)
        setattr(self.pip,field,b[1:])
        
if __name__ == '__main__':
    b = AI_data_pick()
    b.farmer()
    
