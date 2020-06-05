'''robot code manages all program.'''

import DataIn as di
import pipeline as p
import sklar as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class robot(object):
    '''class do manage process'''

    def __init__(self):
        '''inits all objects from pipeline and data in'''

        self.pipeline = p.pipeline()
        self.filmin = di.filmin(self.pipeline)

    def bt(self):
        '''makes brigrt and take operation'''

        #self.bright = di.bright_frame(self.pipeline)
        self.tl = di.line_catcher2(self.pipeline)
        

    def bt_run(self):
        '''runs bt objects'''

        try:
            self.filmin.stp()
        except: print('problem z klatką')
        #self.bright.run()
        self.tl.run()
        
    def dn(self):
        '''normalize data'''

        di.data_normalizer(self.pipeline)
        
    def line_ploter(self):
        '''plots lines'''

        di.line_plotter(self.pipeline)

    def line_ploter2(self):
        '''plots line in 1d style'''

        di.line_plotter_one(self.pipeline)

    def plotter(self, i):
        '''plots lines to animation'''

        plt.clf()
        self.bt_run()
        self.dn()
        sk.sklar(self.pipeline.e_line,self.pipeline.c_line,2,self.pipeline)
        x  = plt.plot(self.pipeline.c_line,'r-', label = 'c')
        y = plt.plot(self.pipeline.l_line,'g-', label = 'l')
        z = plt.plot(self.pipeline.r_line,'b-', label = 'r')
        q = plt.plot(self.pipeline.t_line,'y-', label = 't')
        w = plt.legend()

        return(x,y,z,q,w)
        
        
    def worker(self):
        '''do things'''

        self.bt()
        self.bt_run()
        self.dn()
        self.pipeline.copy_line()
        fig1 = plt.figure()
        ani = animation.FuncAnimation(fig1,self.plotter, save_count = 4*576)
        plt.show()

def first_proc():
    '''first proces to do something'''

    a = robot()
    a.bt()
    a.bt_run()
    a.dn()
    a.pipeline.copy_line()
    for i in range(300):
        a.bt_run()
        a.dn()
        sk.sklar(a.pipeline.e_line,a.pipeline.c_line,2,a.pipeline)
        a.line_ploter2()
        
def loc_ploter():
    '''generator of frames to aplications'''

    pass

    
if __name__ == '__main__':

    a = robot()
    a.worker()
    #first_proc()

    #a = robot()
    #a.bt()
    #a.bt_run()
    #a.line_ploter2()
    #a.dn()
    #b = sk.sklar(a.pipeline.c_line,a.pipeline.l_line,1,a.pipeline)
    #a.line_ploter2()
