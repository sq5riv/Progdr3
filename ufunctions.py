'''Utlility functions set'''
from pipeline import pipeline as pipeline
from DataIn import take_for_AI as taker
from statistics import median

class over_UF(object):
    '''mother object for utility function'''

    def __init__(self, pipeline):
        '''inits pipeline and call rest of all'''

        self.UF = None
        self.pipeline = pipeline
        self.check()
        self.run()
        

    def check(self):
        '''checke data from pipeline'''
        pass
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
        print(dle)
        dle = [x  for x in dle if(x>mx/3)]
        print(dle)
        self.UF = median(dle)

class y_UF(over_UF):
    pass

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
        self.UF = cent - pos

        
if __name__=='__main__':
    z = pipeline()
    taker(z)
    a = x_UF(z)
    print(a.show())
