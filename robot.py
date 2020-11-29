'''robot code manages all program.'''

import DataIn as di
import pipeline as p
#import sklar as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fuzzy_ops2 import fuzzy_set as fs
import fuzzy_ops2 as fop2

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
        self.start = di.starter(self.pipeline)
        
    def bt_rst(self):
        '''reset read of data'''

        del(self.filmin)
        self.pipeline.GO = True
        self.filmin = di.filmin(self.pipeline)
        print('filmin reset')
        
    def bt_run(self):
        '''runs bt objects'''

        '''try:
            self.filmin.stp()
        except:
            d = {'GO':False}
            self.pipeline.set_data(**{'GO':False})
            print('problem z klatką')
        #self.bright.run()'''
        self.filmin.stp()
        self.tl.run()
        self.start.run()
        
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

    def new_plotter(self,i,j):
        '''plots new fuzzy set'''

        plt.clf()
        try:
            self.bt_run()
            self.dn()
            #x = [i for i in range(len(self.pipeline.c_line))]
            #alpha_levels = [0.1,0.3,0.5,0.8,0.9]
            c = fs(self.pipeline.get_domain(), self.pipeline.c_line, self.pipeline.alpha_levels)
            #print(id(c.get_Acuts()))
            #e = fs(x, self.pipeline.e_line, alpha_levels)
            #print(id(e.get_Acuts()))
            #e = c.fuzzy_sub(d, fop2.T_min)
            #print(e.get_Acuts)
           # print(i,j)
            f = self.pipeline.e_fuzz.fuzzy_sub(c, fop2.T_my_drastic,tn_param=j)
            name = 'sub sklar p = ' + str(j)
            dat = {name:f}
            c.plot(show = False, **dat)
            plt.plot(c.get_function(), 'r-', label='c')
            plt.plot(self.pipeline.e_fuzz.get_function(), 'g-', label='e')
            #print(kwarg)
        except: plt.close(self.fig1)

    def newer_plotter(self, i):
        '''do things'''

        plt.clf()
        try:
            loc = self.j[0]
        except: plt.close(self.fig1)
        try:
            self.bt_run()
        except:
            print('error')
            self.bt_rst()
            self.bt()
            self.bt_run()
            try:
                self.j.pop(0)
            except:  plt.close(self.fig1)
            
        #print(i)
        self.dn()
        c = fs(self.pipeline.get_domain(), self.pipeline.c_line, self.pipeline.alpha_levels)
        f = self.pipeline.e_fuzz.fuzzy_sub(c, fop2.T_yager,tn_param=loc)
        name = 'sub yager p = ' + str(loc)
        dat = {name:f}
        c.plot(show = False, **dat)
        plt.plot(c.get_function(), 'r-', label='c')
        plt.plot(self.pipeline.e_fuzz.get_function(), 'g-', label='e')
            
    def worker(self):
        '''do things'''

        self.bt()
        self.bt_run()
        self.dn()
        self.pipeline.copy_line()
        fig1 = plt.figure()
        ani = animation.FuncAnimation(fig1,self.plotter)
        plt.show()

    def worker2(self):
        '''do other things'''

        self.bt()
        self.bt_run()
        self.dn()
        self.pipeline.copy_line()
        self.pipeline.e_fuzz = fs(self.pipeline.get_domain(), self.pipeline.e_line, self.pipeline.alpha_levels)
        self.fig1 = plt.figure()
        ani = animation.FuncAnimation(self.fig1, self.new_plotter, fargs = (0.8,))
        plt.show()

    def worker3(self):
        '''do other things'''

        self.bt()
        self.j = [0,0.1,1]
        self.bt_run()
        self.dn()
        self.pipeline.copy_line()
        self.pipeline.e_fuzz = fs(self.pipeline.get_domain(), self.pipeline.e_line, self.pipeline.alpha_levels)
        self.fig1 = plt.figure()
        ani = animation.FuncAnimation(self.fig1, self.newer_plotter, interval =10)
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

def onego():
    '''goes with one tnorm'''

    a = robot()
    a.worker2()

def multigo():
    '''goes with family of tnorm'''

    a = robot()
    a.worker3()
    
if __name__ == '__main__':

    multigo()
    #first_proc()

    #a = robot()
    #a.bt()
    #a.bt_run()
    #a.line_ploter2()
    #a.dn()
    #b = sk.sklar(a.pipeline.c_line,a.pipeline.l_line,1,a.pipeline)
    #a.line_ploter2()
