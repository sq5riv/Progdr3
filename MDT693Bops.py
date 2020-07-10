from pyvisa.constants import  StopBits, Parity
import pyvisa
from time import sleep
import re
import random

class xyz(object):
    '''class to operate 3-axis piezo controller'''

    def __init__(self):
        '''inits connection'''

        rm = pyvisa.ResourceManager()
        
        try:
            #print(rm.resources_list)
            self.inst = rm.open_resource('ASRL3::INSTR', read_termination='\r')
            self.baud_rate = 115200
            self.inst.data_bits = 8
            self.inst.stop_bits = StopBits.one
            self.inst.parity = Parity.none
            self.inst.flowcontroll = 0
            self.inst.read_termination = '\r\n'
            print('connection established')
        except:
            print('connection error')
        self.pat = re.compile('(\d{1,3}).(\d\d)|0|(\d{1,3})')#generate pattern for search data
        
    def patient(self, command):
        '''talk to device and listen dumb answers'''

        goon = True
        ohit = False
        while goon: 
            try:
                x = self.inst.query(command)
                sleep(0.01)
            except pyvisa.errors.VisaIOError:
                pass
            try:
                if x != command and ohit == True: goon = False #stop loop value
                if x != command: ohit = True
            except UnboundLocalError:
                pass
        
        #print('pattern = ',pat.search(x).group())
        #print(x)
        try:
            retval = self.pat.search(x).group()
        except AttributeError:
            print(x)
            retval = self.patient(command)
            
        return retval
    
    def state(self):
        '''reads state of device'''

        x = float(self.patient('xvoltage?'))
        y = float(self.patient('yvoltage?'))
        z = float(self.patient('zvoltage?'))
        
        return(x,y,z)

    def set(self, k, val):
        '''sets value on some channel of piezo-controller'''

        if not isinstance(val, float):
            raise ValueError ('set function needs float')
        if k == 'x':
            k = 'xvoltage='
        elif k == 'y':
            k = 'yvoltage='
        elif k == 'z':
            k = 'zvoltage='
        command = k + str(val)
        self.inst.write(command)
    def close(self):
        '''close connection'''

        self.inst.close()
class rnd(object):
    '''make random opertions'''

    def __init__(self):

        self.down = 0.00
        self.up = 150.00
        
    
    def run(self):
        '''returns random number from 0,00 to 150,00'''
        return random.uniform(self.down,self.up)
    
def rr():
    a =xyz()
    c = rnd()
    a.set('x',c.run())
    a.set('y',c.run())
    a.set('z',c.run())
    a.close()
    
def look():
    a = xyz()
    for i in range(1000):
        print(i, a.state())
    a.close()
    
if __name__ =='__main__':
    look()

