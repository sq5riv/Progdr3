

class sklar(object):
    '''sklar parametric t-norm implementation'''

    def __init__(self, x,y,p, pipeline):

        self.x = x
        self.y = y
        self.p = p
        self.pipeline = pipeline
        self.val = []

        if p ==float('-inf'):
            self.one()
        elif p > float('-inf') and p < 0:
            self.two()
        elif p == 0:
            self.three()
        elif p > 0 and p <float('inf'):
            self.four()
        elif p == float('inf'):
            self.five()

        self.save()
        
    def one(self):

        for i  in  range(len(self.x)):
            self.val.append(min(self.x[i],self.y[i]))
                    
    def two(self):

        for i in range(len(self.x)):
            try:
                self.val.append((self.x[i]**self.p+self.y[i]**self.p-1)**(1/self.p))
            except ZeroDivisionError:
                self.val.append(0.0)
            
    def three(self):

        for i in range(len(self.x)):
            self.val.append(self.x[i]*self.y[i])
            
    def four(self):

        for i in range(len(self.x)):
            self.val.append((max(0,self.x[i]**self.p+self.y[i]**self.p-1))**(1/self.p))
            
    def five(self):
        for i in range(len(self.x)):
            if self.x[i] == 1: self.val.append(self.y[i])
            elif self.y[i] == 1: self.val.append(self.x[i])
            else: self.val.append(0)
            
    def save(self):
        self.pipeline.t_line = self.val
    
