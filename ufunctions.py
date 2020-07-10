'''Utlility functions set'''
from pipeline import pipeline as pipeline
from DataIn import take_for_AI as taker
from statistics import median
from functools import reduce
from statistics import mean

class over_UF(object):
    '''mother object for utility function'''

    def __init__(self, pipeline):
        '''inits pipeline and call rest of all'''

        self.UF = None
        self.pipeline = pipeline
        
        

    def check(self):
        '''checke data from pipeline'''
        pass

    def do(self):
        '''organizes all work'''

        self.check()
        self.run()

        return self.show()
    
    def run(self):
        '''make work'''
        pass
    def show(self):
        '''shows result of utility function'''
        return self.UF

class x_UF(over_UF):
    '''Calculates value of UF in x direction'''

    def check(self):
        '''takes data from pipeline'''

        self.datin = list(self.pipeline.AI_c_line)
        self.datin = [a-b for a,b in zip(self.datin, self.datin[1:]+self.datin[:1])]

    def run(self):
        '''calculates value of UF'''

        dle = [0]
        lv = 1
        for i in self.datin:
            if i > 0 and lv > 0: dle[-1]+=1
            elif i< 0 and lv < 0: dle[-1]+=1
            elif i > 0 and lv < 0:
                dle.append(0)
                lv = 1
            elif i < 0 and lv > 0:
                dle.append(0)
                lv = -1
        mx = max(dle)
        #print(dle)
        dle = [x  for x in dle if(x>mx/3)]
        #print(dle)
        self.UF = abs(median(dle)-50)

class y_UF(over_UF):
    '''Calculates y utility finction'''

    def check(self):
        '''takes data from pipeline'''

        self.datin_h = list(self.pipeline.h_line)
        self.datin_c = list(self.pipeline.AI_c_line)[40:-40]
        self.datin_r = list(self.pipeline.AI_r_line)[40:-40]
        self.datin_l = list(self.pipeline.AI_l_line)[40:-40]

    def run(self):
        '''calculates value of UF'''

        c = self.datin_c.index(min(self.datin_c))
        r = self.datin_r.index(min(self.datin_r))
        #l = self.datin_l.index(min(self.datin_l))

        cr = abs(c-r)
        #cl = c-l
        
        self.UF =cr#+cl  

class z_UF(over_UF):
    '''calculates value of UF in z direction'''
    
    def check(self):
        '''takes data from pipeline'''

        self.datin = list(self.pipeline.AI_c_line)
        self.datin = self.datin[30:-30]

    def run(self):
        '''calculates value of UF'''

        
        cent = len(self.datin)/2
        pos = self.datin.index(min(self.datin))
        self.UF = abs(cent - pos)

class UF_sum(over_UF):
    '''sums x_UF, y_UF, z_UF'''

    def run(self):
        '''calculates UF'''

        tmp_x = x_UF(self.pipeline)
        tmp_y = y_UF(self.pipeline)
        tmp_z = z_UF(self.pipeline)
        x = tmp_x.do()
        y = tmp_y.do()
        z = tmp_z.do()
        #print(x,y,z)
        self.UF = x+y+z
        print(x,y,z,self.UF)
        
if __name__=='__main__':
    z = pipeline()
    taker(z)
    a = UF_sum(z)
    print(a.do())
