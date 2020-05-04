# Filename: main_window.py

"""Main Window-Style application."""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
import time

__version__ = '0.2'
__author__ = 'Michał Kalenik'




class window(QMainWindow):
    """Main Window."""
    def __init__(self, boss):

        """Initializer."""
        
        super().__init__(None)
        #Window properties
        self.setWindowTitle('Bisektor')
        self.setFixedSize(235, 80)
        
        #Boss object
        self.boss = boss
        
        
        #General Layout
        self._general_layout = QGridLayout()
        self._central_widget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self.ll_button = QPushButton('<-')
        self.lr_button = QPushButton('->')
        self.rl_button = QPushButton('<-')
        self.rr_button = QPushButton('->')
        #self.ll_button.setCheckable(True)
        #self.lr_button.setCheckable(True)
        #self.rl_button.setCheckable(True)
        #self.rr_button.setCheckable(True)
        self._general_layout.addWidget(QPushButton('Lewy'),0,0,1,2)
        self._general_layout.addWidget(QPushButton('Prawy'),0,2,1,2)
        self._general_layout.addWidget(self.ll_button,1,0)
        self._general_layout.addWidget(self.lr_button,1,1)
        self._general_layout.addWidget(self.rl_button,1,2)
        self._general_layout.addWidget(self.rr_button,1,3)
        self._central_widget.setLayout(self._general_layout)
        #Variables
        self._left = 255
        self._right = 300
        self.l = 0
        self.r = 0
        #Connect signals
        #self.ll_button.pressed.connect(lambda x: self._l_go(-1))
        #self.lr_button.pressed.connect(lambda x: self._l_go(1))
        #self.rl_button.pressed.connect(lambda x: self._r_go(-1))
        #self.rr_button.pressed.connect(lambda x: self._r_go(1))
        #self.ll_button.released.connect(lambda x: self.r_go(0))
        #self.lr_button.released.connect(lambda x: self._l_go(0))
        #self.rl_button.released.connect(lambda x: self._r_go(0))
        #self.rr_button.released.connect(lambda x: self._r_go(0))

        self.ll_button.pressed.connect(self._ll)
        self.lr_button.pressed.connect(self._lr)
        self.rl_button.pressed.connect(self._rl)
        self.rr_button.pressed.connect(self._rr)
        



    def _l_go(self, n):
        '''left go variable set'''

        print('in ll')        
        if n == 0 and self.l != 0:
            self.boss.l_go(n)
            self.l = 0
        if n != 0:
            self.boss.l_go(n)

    def _r_go(self, n):
        '''left go variable set'''

        if n == 0 and self.r != 0:
            self.boss.r_go(n)
            self.r = 0
        if n != 0:
            self.boss.r_go(n)        
        
    def _ll(self):
        '''left goes left'''

        self.boss.update_left(-1)

    def _lr(self):
        '''left goes right'''

        self.boss.update_left(1)

    def _rl(self):
        '''right goes left'''

        self.boss.update_right(-1)

    def _rr(self):
        '''right goes fight'''

        self.boss.update_right(1)

    def closeEvent(self, event):
        '''sets stop in post_office'''

        self.boss.set_go()
        print('stop')
        
    def showme(self):
        '''raturns data'''

        return(self._left, self._right)


        
def main(boss):
    '''Main function'''

    
    app = QApplication(sys.argv)
    win = window(boss)
    win.show()

    sys.exit(app.exec_())
    

    


if __name__ == '__main__':
    main('a')
