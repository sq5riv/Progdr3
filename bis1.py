import numpy as np
import cv2
import matplotlib.pyplot as plt


__version__ = '0.2'
__author__ = 'Michał Kalenik'
OPENCV_VIDEOIO_DEBUG=1
OPENCV_LOG_LEVEL='debug'

class worker(object):
    '''worker make videao capture and ads lines to it.'''

    def __init__(self, left, right, r,g,b, boss):
        '''Initialize capture and make main procedure. '''
        
        #logging.info('In main')
        #variables
        self.left = 0
        self.right = 0
        self.boss = boss
        self.horizonal = 288
        self.center_x = 384
        self.left_x = 0
        self.right_x = 0
        self.color = [0,0,0] #['b','g','r']
        #shape x = 768, y = 576 #nope 640:480 # nope first was better. !
        #init sets
        self.set_lr(left, right)
        self.set_col(r,g,b)
        #start main loop
        self.aaa_loop()

        
    def aaa_loop(self):
        '''main loop of object'''

        print('wait  for camera if too long restart computer')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print('camera on')
        print('If there is no image, check lightsource or restart program')
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        
        #while(cap.isOpened()):
        while(self.boss.wego()):
            #a = False
            #print('in1')
            ret, self.frame = cap.read()
            self.x, self.y, self.z = self.frame.shape
            #print(self.x, self.y)
            self.horizonal = int(self.x/2)
            self.center_x = int(self.y/2)
            #print(self.frame[self.center_x][self.horizonal])
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.left == 0:
                self.boss.set_lr(self.center_x - 100, self.center_x + 100)
                #self.find_lr()            
                
            #print(type(self.boss.show_r()))
            self.boss.update_left(self.boss.show_l())
            self.boss.update_right(self.boss.show_r())
            
            self.horizontal_adder(self.horizonal)
            l, r = self.boss.show_me()
            #print(l,r)
            self.set_lr(l,r)
            self.vert_adder(self.left)
            self.vert_adder(self.right)
            self.vert_short_adder(self.left_x,30)
            self.vert_short_adder(self.center_x,30)
            self.vert_short_adder(self.right_x,30)
            
            #print(self.frame)
            cv2.imshow('frame',self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        cap.release()
        cv2.destroyAllWindows()

    def find_lr(self):
        '''finds left and right side of gauge block and call self.set_lr'''

        gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        sum_list = np.sum(gray, axis = 0)
        plt.plot(sum_list)
        plt.show()
        print(self.x, self.y)
        #div_list = [sum_list[i]-sum_list[i+1] for i in range(len(sum_list)-2)]
        
        div_list = []
        for i in range(len(sum_list)-1):
            div_list.append(abs(int(sum_list[i])-int(sum_list[i+1])))

        plt.plot(div_list)
        plt.show()
        xh = int(self.y/2)
        #print(len(div_list))
        left = div_list.index(max(div_list[50:xh]))
        right = div_list.index(max(div_list[xh:-50]))+xh

        self.set_lr(left,right)
        self.boss.set_lr(left, right)
        #plt.plot(div_list)
        #plt.show()
        
                          
    def horizontal_adder(self, y):
        '''ads horizontal line to frame'''
        
        np.put(self.frame, [range(self.y*y*3,self.y*(y+1)*3)], self.color)

        
    def set_lr(self, left, right):
        '''sets self.left and self.right'''

        self.left = left
        self.right = right
        self.center_x = int((left+right)/2)
        self.left_x = int(self.center_x+left-right)
        self.right_x = int(self.center_x-left+right)

    def set_col(self, r,g,b):
        '''sets color of line'''

        self.color = [b,g,r]

    def set_text(self):
        '''sets text on frame'''

        cv2.PutText(self.frame, 'Ala ma kota', [100*self.x*100*3], self.color)
        
    def vert_adder(self, x):
        '''ads vertical line to frame'''
        
        np.put(self.frame, [range(x*3, self.y*self.x*3,self.y*3)], self.color[0])
        np.put(self.frame, [range(x*3+1, self.y*self.x*3,self.y*3)], self.color[1])
        np.put(self.frame, [range(x*3+2, self.y*self.x*3,self.y*3)], self.color[2])

        
    def vert_short_adder(self, x, y):
        '''ads short vertical line fo frame'''

        xh = int(self.x/2)
        np.put(self.frame, [range(x*3+self.y*3*(xh-y),x*3+self.y*3*(xh+y),self.y*3)],self.color[0])
        np.put(self.frame, [range(x*3+1+self.y*3*(xh-y),x*3+self.y*3*(xh+y),self.y*3)],self.color[1])
        np.put(self.frame, [range(x*3+2+self.y*3*(xh-y),x*3+self.y*3*(xh+y),self.y*3)],self.color[2])
    
if __name__ == '__main__':
    b = worker(270,490,255,255,255,'aaa')
    
