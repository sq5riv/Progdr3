import threading
import bis1
import window
import time
import sys


__version__ = '0.2'
__author__ = 'Michał Kalenik'

class post_office(object):
    '''post office contains data for worker'''


    def __init__(self):
        '''init left and right variables'''

        self.left = 0
        self.right = 0
        self.l_go = 0
        self.r_go = 0
        
        self.go = True

    def l_go(self, n):
        '''sets left go variable'''

        self.l_go = n

    def r_go(self, n):
        '''sets left go variable'''

        self.r_go = n
        

    def set_go(self):
        '''sets go to false'''

        self.go = False
        
    def set_lr(self, left, right):
        '''sets left and right'''

        self.left = left
        self.right = right
        
    def show_l(self):
        '''returns l_go'''

        return self.l_go
    
    def show_me(self):
        '''returns left and right'''

        return self.left, self.right

    def show_r(self):
        '''returns l_go'''

        return self.r_go

    def update_left(self, n):
        '''updates left by n'''

        self.left = self.left + n

    def update_right(self, n):
        '''updates right by n'''

        self.right = self.right + n

    def wego(self):
        '''returns value of go'''

        return self.go
        
if __name__ == '__main__':

    print('Starting')
    boss = post_office()
    wait = 0 #sec
    print('Wait for %s seconds' %wait)
    time.sleep(wait)
    print('threads declare')
    window = threading.Thread(target = window.main, args=(boss, ))
    worker = threading.Thread(target = bis1.worker, args = (0,0,255,255,255,boss))
    print('Start window thread')
    window.start()
    print('Start bisector thread')
    worker.start()
    
