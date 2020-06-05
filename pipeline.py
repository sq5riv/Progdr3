import numpy as np
import copy

class pipeline(object):
    '''class to save data for modificators etc.'''

    def __init__(self):
        '''init start all fields'''

        #bisektor
        self.left = 250
        self.right = 450
        self.taken_width = 30
        #kolor
        self.color = 'no' #'red','green'
        self.brightner = 0
        self.bright_on = 'on' #'on'/'off'
        self.frame_mask = 'no'
        #list of modifications
        self.lom = []
        #info for live actions:
        self.GO = True
        #info for filmin
        self.source = 'redtest1.avi' #'greentest.avi' #'redtest1.avi' #'None' #None is for camera. Or use filepath
        self.num_to_save = 200
        self.save_path = 'gru.avi'
        #frame shape
        self.width = 0
        self.height = 0
        #frame data
        self.frame = 0
        #lines
        self.c_line = 0
        self.l_line = 0
        self.r_line = 0
        self.e_line = 0 # ethalon line is line for inter frame ops.
        self.t_line = 0 #tnorm of lines
        #reference_line
        self.ref_line = 0
        #list of maches
        self.list_of_maches = []
        #params of cut
        self.cut = [20,-20,10,250]
        #params of smooth
        self.order = 8
        self.Wn = 0.125
        self.btype = 'low' #type of filter
        self.cut_freq = 100
        
    def copy_line(self):
        '''copy c_line to e_line'''

        self.e_line = copy.copy(self.l_line)
        
    def get_frame(self):
        '''returns copy of frame'''

        return self.frame.copy()

    def set_data(self, **kwargs):
        '''sets data in pipeline'''

        for k,v in kwargs.items():
            setattr(self, k,v)

    def set_frame(self, frame):
        '''sets frame'''

        self.frame = frame

    def set_go(self):
        '''sets GO to false'''
        self.GO = False

    def set_mask(self):
        '''sets mask frame to brightner'''

        if self.color == 'green': color = 1
        elif self.color == 'red': color = 2
        elif self.color == 'blue': color = 0
        a = np.uint8(1)
        p = np.uint8(self.brightner)
        lst = [a,a]
        lst.insert(color,p)
        lst = lst*self.width*self.height
        buff = np.array(lst)
        self.frame_mask = np.ndarray(shape=(self.height,self.width,3), dtype=np.uint8, buffer=buff)
    def show_data(self, val):
        '''returns other types of data for other whos'''
        try:
            return getattr(self, val)
        except NameError: ('No data')
    def update_left(self, n):
        '''update left'''

        self.left +=n

    def update_right(self, n):
        '''update right'''

        self.right += n

