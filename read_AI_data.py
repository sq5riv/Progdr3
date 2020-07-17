'''script to read AI data for train decision tree'''
from sklearn import tree
from sklearn.tree import export_text
import numpy as np
from random import choice

class data_reader(object):

    def __init__(self):

        self.dat_list = []
        with open('AI_dat.txt') as f:
            for line in f:
                self.dat_list.append(line)

    def filter(self):

        self.dat_list2 = []
        for line in self.dat_list:
            #print(line)
            line = line.replace('\n','')
            line = line.split(' ')
            #print(line[1])
            if abs(float(line[1])) <10 and abs(float(line[3]))<10 and abs(float(line[5]))<10:
                self.dat_list2.append(line)
        

    def filter2(self):

        tmpx = ['0']*10
        tmpy = ['0']*10
        tmpz = ['0']*10
        tmpuf = ['0']*10
        X = []
        Y = []
        
        for dat in self.dat_list2:
            tmpx.append(dat[1])
            tmpy.append(dat[3])
            tmpz.append(dat[5])
            tmpuf.append(dat[-1])
            tmpx = tmpx[1:]
            tmpy = tmpy[1:]
            tmpz = tmpz[1:]
            tmpuf = tmpuf[1:]
            load = []
            load.extend(tmpx)
            load.extend(tmpy)
            load.extend(tmpz)
            load.extend(tmpuf)
            
            X.append(load)
            
            if float(dat[1]) > 0.0:
                Y.append('x+')
            elif float(dat[1])<0.0:
                Y.append('x-')
            elif float(dat[3])>0.0:
                Y.append('y+')
            elif float(dat[3]) < 0.0:
                Y.append('y-')
            elif float(dat[5])>0.0:
                Y.append('z+')
            elif float(dat[5])<0.0:
                Y.append('z-')
            else:
                Y.append(0)
        Y = Y[1:]+Y[:1]
        self.X = X
        self.Y = Y

    def tree(self):

        clf = tree.DecisionTreeClassifier()
        
        self.X = np.array(self.X)
        self.Y =np.array(self.Y)
        #print(self.Y)
        self.clf = clf.fit(self.X, self.Y)

    def tree_print(self):

        print(export_text(self.clf))

    def digg_func(self):

        Z = []
        for y in self.Y:
            pom = True
            for z in Z:
                if z ==y: pom = False
            if pom == True: Z.append(y)
        self.Z = Z

    def ret_command(self, datin):
        '''returns next command for execute'''

        dec = self.clf.predict_proba([datin])
        dec2 = []
        for i,v in enumerate(dec[0]):
            if v !=0:
                dec2.append(i)
        ret_index = choice(dec2)

        try:
            retval = self.Z[ret_index]
        except AttributeError: 
            self.digg_func()
            retval = self.Z[ret_index]
            
        return retval
            
## End of objects.
def AI_ops():
    a = data_reader()
    a.filter()
    a.filter2()
    a.tree()
    a.tree_print()
    print(a.clf.proba([[0]*30]))
    
if __name__ =='__main__':
    a = data_reader()
    a.filter()
    a.filter2()
    a.tree()
    #a.tree_print()
    #print(a.dat_list2)
    print(a.ret_command([0]*40))
    
