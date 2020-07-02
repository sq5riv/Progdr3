from pyvisa.constants import  StopBits, Parity
import pyvisa
from time import sleep
import re

class xyz(object):
    '''class to operate 3-axis piezo controller'''

    def __init__(self):
        '''inits connection'''

        rm = pyvisa.ResourceManager()
        
        try:
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
        self.pat = re.compile('(\d{1,3}).(\d\d)|0')#generate pattern for search data
        
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

        x = self.patient('xvoltage?')
        y = self.patient('yvoltage?')
        z = self.patient('zvoltage?')
        
        return(x,y,z)
    
    def close(self):
        '''close connection'''

        self.inst.close()

if __name__ =='__main__':
    a = xyz()
    for i in range(1000):
        print(i, a.state())
    a.close()
