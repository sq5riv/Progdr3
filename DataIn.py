import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
from pipeline import pipeline as pipeline
from scipy import signal

class filmin(object):
	'''Klasa pobieraków danych'''
	
	def __init__(self, pipeline):
		'''inits data'''
		
		self.pipeline = pipeline
		self.source = self.pipeline.show_data('source')
		while self.pipeline.show_data('GO'):
			try:
				self.go()
				break
			except ValueError:
				self.rstart()
				print('FrameError - reset')

	def check(self):
		'''checks if first frame is  ok'''#ok
		
		ret, frame = self.cap.read()
		if ret == False : raise ValueError('Frame is bad')
		else: 
			height, width, z = frame.shape
			sh = {'width' : width, 'height' : height}
			
			sm = np.sum(np.sum(frame, 0),0)
			color = np.argmax(sm)
			valmax = np.max(sm)/height/width/15 #dobrane doświadczalnie 
			sh.update({'brightner':valmax}) 
			if color == 1: sh.update({'color':'green'})
			elif color == 2: sh.update({'color':'red'})
			elif color == 0: sh.update({'color':'blue'})
			self.pipeline.set_data(**sh)
			self.pipeline.set_mask()
			
	
	def check2(self):
		'''checks all attributes of film'''#ok
		
		lst = ['CV_CAP_PROP_POS_MSEC', 'CV_CAP_PROP_POS_FRAMES', 'CV_CAP_PROP_POS_AVI_RATIO',
		'CV_CAP_PROP_FRAME_WIDTH', 'CV_CAP_PROP_FRAME_HEIGHT', 'CV_CAP_PROP_FPS', 'CV_CAP_PROP_FOURCC',
		'CV_CAP_PROP_FRAME_COUNT', 'CV_CAP_PROP_FORMAT', 'CV_CAP_PROP_MODE', 'CV_CAP_PROP_BRIGHTNESS',
		'CV_CAP_PROP_CONTRAST', 'CV_CAP_PROP_SATURATION', 'CV_CAP_PROP_HUE', 'CV_CAP_PROP_GAIN', 
		'CV_CAP_PROP_EXPOSURE', 'CV_CAP_PROP_CONVERT_RGB', 'CV_CAP_PROP_WHITE_BALANCE_U', 'CV_CAP_PROP_WHITE_BALANCE_V',
		'CV_CAP_PROP_RECTIFICATION', 'CV_CAP_PROP_ISO_SPEED', 'CV_CAP_PROP_BUFFERSIZE']
		for i, val in enumerate(lst):
			print(val, self.cap.get(i))
				
	def cio(self):
		'''returns info about cap.isOpened'''#ok
		
		return self.cap.isOpened()
		
	def end(self):
		'''finish work'''#ok
		
		self.cap.release()
		cv2.destroyAllWindows()
		
	def go(self):
		'''starts Videocapture and check it'''
		
		if self.source == 'None':
			self.cap = cv2.VideoCapture(0)#, cv2.CAP_DSHOW)
		else: 
			self.cap = cv2.VideoCapture(self.source)
		
		self.check()
		self.stp()
		
		
	def rstart(self):
		'''close cap'''
		
		self.cap.release()
			
	def show(self):
		'''returns single frame'''
		
		if self.cap.isOpened():
			ret, frame = self.cap.read()
			if(type(frame)==type(None)):raise ValueError('End of frames')
			return frame
		else: raise ValueError("No more frames")

	def stp(self):
		'''sets frame in pipeline'''
		
		if self.cap.isOpened():
			ret, frame = self.cap.read()
			if ret == False: 
				d = {'GO':False}
				self.pipeline.set_data(**d) 
				raise ValueError('End of frames')
			d = {'frame':frame}
			self.pipeline.set_data(**d)
		else: 
			self.pipeline.set_data({'GO':False}) 
			raise ValueError("No more frames")
		
class overframe(object):
	'''Klasa matka klatek z przeróbkami'''
	
	def __init__(self, pipeline):
		'''init makes all things with taken frame'''
		
		self.pipeline = pipeline
		#self.frame = self.pipeline.show_data('frame')
		#self.check()
		#self.modyficator()
		#self.run()
	
	def check(self):
		'''checks what its need'''
		
		pass
	
	def modyficator(self):
		'''modifies frame with data from pipeline'''
		
		pass
			
	def run(self):
		'''runs object'''
		
		self.frame = self.pipeline.show_data('frame')
		self.check()
		self.modyficator()
		
		
	def show(self):
		'''returns modified frame'''
		
		return self.frame.copy()			
			
class bright_frame(overframe):
	'''one color is brightend'''
	

	def check(self):
		'''checks pipeline for needed values'''
		
		
		self.go = self.pipeline.show_data('bright_on')
		self.brightner = self.pipeline.show_data('brightner')
		self.frame2 = self.pipeline.show_data('frame_mask')
		
	def modyficator(self):
		'''makes one color brights than others'''
		
		if self.go == 'on':
			
			self.frame = cv2.multiply(self.frame,self.frame2)
			self.pipeline.set_frame(self.frame)
								
class bis_frame(overframe):
	
	def check(self):
		'''checks left and right bisector value'''
		
		self.left = self.pipeline.show_data('left')		
		self.right = self.pipeline.show_data('right')
	
	def horizontal_adder(self, y):
		'''ads horizontal line to frame'''
		
		x = self.pipeline.show_data('width')
		np.put(self.frame, [range(int(x*y*3/2),int(x*y*3/2+x*3))], 255)
		
	def modyficator(self):
		'''adds bisector'''
		
		#print('inmod')
		self.horizontal_adder(self.pipeline.show_data('height'))
		self.vert_adder(self.left)
		self.vert_adder(self.right)
		center = (self.left + self.right)/2
		dev = abs(self.left - self.right)/2
		self.vert_short_adder(center, 30)
		self.vert_short_adder(self.right+dev, 30)
		self.vert_short_adder(self.left-dev,30)

	def vert_adder(self, x):
		'''asd vertical line to frame'''
		
		w = self.pipeline.show_data('width')
		h = self.pipeline.show_data('height')
		np.put(self.frame, [range(x*3, h*(w-1)*3, w*3)], 255)
		np.put(self.frame, [range(x*3+1, h*(w-1)*3, w*3)], 255)
		np.put(self.frame, [range(x*3+2, h*(w-1)*3, w*3)], 255)

        
	def vert_short_adder(self, x, y):
		'''ads short vertical line fo frame'''
		
		w = self.pipeline.show_data('width')
		h = self.pipeline.show_data('height')
		h2 = h/2
		np.put(self.frame, [range(int(x*3+w*3*(h2-y)),int(x*3+w*3*(h2+y)),w*3)], 255)
		np.put(self.frame, [range(int(x*3+1+w*3*(h2-y)),int(x*3+w*3*(h2+y)),w*3)], 255)
		np.put(self.frame, [range(int(x*3+2+w*3*(h2-y)),int(x*3+w*3*(h2+y)),w*3)], 255)
 		
class line_catcher(overframe):
	'''gives three lines center, left and right'''
	
	def check(self):
		'''checks left, fight and color and sets valuer for modificator'''
		
		left = self.pipeline.show_data('left')
		right = self.pipeline.show_data('right')
		self.center = int((left + right)/2)
		dev = abs(left - right)/2
		self.left = left - dev
		self.right = right + dev	
		#print(self.left, self.right)
		color = self. pipeline.show_data('color')
		
		if color == 'green': self.color = 1
		elif color == 'red': self.color = 2
		elif color == 'blue': self.color = 0
		
	def modyficator(self):
		'''catches lines from frame'''
		
		#self.c_line = np.take(self.frame, range(self.center*3+self.color, self.center*3+768*574*3+self.color, 768*3))
		self.c_line = self.taker(self.center)
		self.l_line = self.taker(self.left)
		self.r_line = self.taker(self.right)
		d = {'self.c_line':self.c_line,'self.l_line':self.l_line,'self.r_line':self.r_line}
		self.pipeline.set_data(**d)
		
		#plt.plot(self.c_line)
		#plt.show()
		
	def show(self):
		'''shows data'''
		
		
		return (self.c_line, self.l_line, self.r_line)
		
	def taker(self, x):
		'''takes line from frame'''
		
		w = self.pipeline.show_data('width')
		h = self.pipeline.show_data('height')
		tlist = list(range(int(x*3+self.color),int(x*3+w*h*3+self.color),w*3))
		return np.take(self.frame, tlist)
			
		'''for i in range(768):
			self.c_line = np.take(self.frame, range(i*3+self.color, i*3+768*574*3+self.color, 768*3))
			plt.plot(self.c_line)
			plt.show()'''

class takeliner(object):
	'''class made do organice all cuts.'''
	
	def __init__(self, pipeline, line_catcher):
		'''inits all works'''
			
		self.pipeline = pipeline
		self.line_catcher = line_catcher
	
	def run(self):
		'''changes line fo x,y format'''
		
		self.line_catcher.run()
		self.c, self.l, self.r = self.line_catcher.show()
		self.xy_liner('c')
		self.xy_liner('l')
		self.xy_liner('r')
		d = {'c_line':self.c,'l_line':self.l,'r_line':self.r}
		self.pipeline.set_data(**d)		
		
	def xy_liner(self, dat):
		'''change list of y to xy list'''
		
		y = getattr(self, dat)
		x = [x for x in range(len(y))]
		tmp = [x,y]
		setattr(self, dat, tmp)
		
	def show_c(self):
		'''shows central crosssection'''
		
		return self.c

	def show_l(self):
		'''shows left crosssection'''
		
		return self.l
		
	def show_r(self):
		'''shows right crosssection'''
		
		return self.r

class overliner(object):
	'''class over all line operations'''
	
	def __init__(self, pipeline, w_line):
		'''init'''
		
		self.pipeline = pipeline
		self.w_line = w_line
		
	def check(self):
		'''checks params in pipeline'''
		
		pass
		
	def get_line(self):
		'''get_line'''
		
		self.line = getattr(self.pipeline, self.w_line).copy()
		
	def modyficator(self):
		'''make modifications'''
		
		pass
	
	def run(self):
		'''runs object with one data box'''
		
		self.get_line()
		self.check()
		self.modyficator()
		self.set_line()
		
	def set_line(self):
		'''sets line'''
		
		setattr(self.pipeline,self.w_line,self.line)
		
	def show(self):
		'''shows line'''
		
		pass
		
class cut(overliner):
	'''cuts lines'''
	
	def check(self):
		'''checks cut vals'''
		
		self.cut = self.pipeline.cut
		self.len = len(self.line[1])
			
	def modyficator(self):
		'''cuts some data'''
		
		tmp = [[],[]]
		for p in range(len(self.line[1])):
			if self.line[0][p]<self.cut[0]: continue
			elif self.line[0][p]>self.len + self.cut[1]: continue
			elif self.line[1][p]<self.cut[2]:continue
			elif self.line[1][p]>self.cut[3]:continue
			else: 
				tmp[0].append(self.line[0][p])
				tmp[1].append(self.line[1][p])
		self.line = tmp
		'''tmp = []
		for p in self.line:
			x, y = p
			if x < self.cut[0]: continue
			elif x > self.len+self.cut[1]: continue
			elif y < self.cut[2]: continue
			elif y > self.cut[3]: continue
			else: tmp.append(p)
		self.w_frame = tmp'''

class smoother(overliner):
	'''smooth line'''
	
	def __init__(self, pipeline, w_line):
		'''initialization'''

		self.pipeline = pipeline
		self.w_line = w_line		
		self.order = self.pipeline.order
		self.Wn = self.pipeline.Wn
		self.btype = self.pipeline.btype
		self.cut_freq = self.pipeline.cut_freq
		self.b, self.a = signal.bessel(self.order, self.Wn, self.btype)
		
	def modyficator(self):
		'''smooth lone'''
		
		self.line[1] = signal.filtfilt(self.b,self.a, self.line[1], padlen=30)
						
class play_film(object):
	'''plays film'''
	
	def __init__(self, pipeline):
		'''inits frame'''
		
		self.pipeline = pipeline
		self.get_frame()


	def get_frame(self):
		'''gets new frame from pipeline'''
		
		self.frame = self.pipeline.show_data('frame')
	
	def run(self):
		'''shows image'''
		
		#print('in run')
		cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('frame', 800,800)
		
		while self.pipeline.show_data('GO'):
			
			try:
				cv2.imshow('frame',self.frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			except ValueError:
				print('nmf')
				break		
				
	def start(self):
		'''starts work of object'''
		threading.Thread(target=self.run).start()
		
class save_film(object):
	'''class to save film'''
	
	def __init__(self, pipeline):
		'''saves frame by frame to file'''
		
		
		self.pipeline = pipeline
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		w = self.pipeline.show_data('width')
		h = self.pipeline.show_data('height')
		self.out = cv2.VideoWriter(self.pipeline.save_path,fourcc,30.0,(w,h))
		
	def save(self):
		'''saves one frame'''
		
		if self.pipeline.show_data('num_to_save')>self.i: 
				self.out.write(self.pipeline.get_frame())
				
				
				
	def end(self):
		'''ends work'''
				
		self.out.release()

class switch(object):
	'''switch frames from pipeline to pipeline'''
	
	def __init__(self, pip_a, pip_b):
		'''copies frame from pipeline a to pipeline b'''
		
		self.pip_a = pip_a
		self.pip_b = pip_b
	
	def doit(self):
		'''do work'''
		
		self.pip_b.set_frame(self.pip_a.get_frame())

def take_one_cut():
    '''simpla working program to take first cut'''
    a = pipeline()
    print('pipeline ok')
    b = filmin(a)
    print('filmin ok')
    bright = bright_frame(a)
    bis = bis_frame(a)
    play = play_film(a)
    tl = takeliner(a,line_catcher(a))
    sm = smoother(a, 'c_line')
    ct = cut(a, 'c_line')
    msl = smoother(a, 'l_line')
    ctl = cut(a, 'l_line')
    smr = smoother(a, 'r_line')
    ctr = cut(a, 'r_line')
    #ovl = overliner(a, 'c_line')
	
    for i in range(500):
        try:
            b.stp()
        except: break
        bright.run()
        tl.run()
        bis.run()
        play.get_frame()
        sm.run()
        ct.run()
        msl.run()
        ctl.run()
        smr.run()
        ctr.run()
		
		
        #print(a.c_line)	
    plt.plot(a.c_line[0],a.c_line[1], '.')	
    plt.plot(a.l_line[0],a.l_line[1], '.')	
    plt.plot(a.r_line[0],a.r_line[1], '.')
    plt.show()
    a.GO = False	
    b.end()
def take_o_cut():
                '''takes other cut'''
                pass
if __name__=='__main__':
    take_one_cut()
    

#sprawdzic linechether
#napisać plotter
#napisać phaser
#napisać fuzzy phaser
#napisać fuzy phaser2


	
	
	
