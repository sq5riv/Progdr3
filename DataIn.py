import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

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
			valmax = np.max(sm)
			sh.update({'brightner':331361280/valmax/6.21}) #6.21 dobrane doświadczalnie.
			if color == 1: sh.update({'color':'green'})
			elif color == 2: sh.update({'color':'red'})
			elif color == 0: sh.update({'color':'blue'})
			self.pipeline.set_data(**sh)
			self.pipeline.set_frame()
			
	
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

class overframe(object):
	'''Klasa matka klatek z przeróbkami'''
	
	def __init__(self, frame, pipeline):
		'''init makes all things with taken frame'''
		
		self.pipeline = pipeline
		self.frame = frame
		self.check()
		self.modyficator()
	
	def check(self):
		'''checks what its need'''
		
		pass
	
	def modificator(self):
		'''modifies frame with data from pipeline'''
		
		pass
		
	def show(self):
		'''returns modified frame'''
		
		return self.frame.copy()			
			
class bright_frame(overframe):
	'''one color is brightend'''
	

	def check(self):
		'''checks pipeline for needed values'''
		
		
		self.go = self.pipeline.show_data('bright_on')
		self.color = self.pipeline.show_data('color')
		self.brightner = self.pipeline.show_data('brightner')
		self.width = self.pipeline.show_data('width')
		self.height = self.pipeline.show_data('height')
		self.frame2 = self.pipeline.show_data('frame_mask')
		
	def modyficator(self):
		'''makes one color brights than others'''
		
		if self.go == 'on':
			
			print(type(self.frame),type(self.frame2))
			self.frame = cv2.multiply(self.frame,self.frame2)
						
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
		
		plt.plot(self.c_line)
		plt.show()
		
	def show(self):
		'''shows data'''
		
		return (self.c_line, self.l_line, self.r_line)
		
	def taker(self, x):
		'''takes line from frame'''
		
		w = self.pipeline.show_data('width')
		h = self.pipeline.show_data('height')
		return np.take(self.frame, range(int(x*3+self.color),int(x*3+w*h*3+self.color),w*3))
			
		'''for i in range(768):
			self.c_line = np.take(self.frame, range(i*3+self.color, i*3+768*574*3+self.color, 768*3))
			plt.plot(self.c_line)
			plt.show()'''

class mod_clip(overframe):
	'''plays film'''
	
	def check(self):
		'''checks for list of modifications'''
		
		self.lom = self.pipeline.show_data('lom')

	
	def modyficator(self):
		'''makes all modifies'''
			
		self.loo = []
		self.loo.append(self.frame.show())
		
		for num in range(len(self.lom)-1):
			mod = getattr(self.lom[num](self.loo[-1],self.pipeline), 'show')
			self.loo.append(mod())
			
	
	def show(self):
		'''shows modified frame'''
		
		return self.loo[-1]

class takeliner(object):
	'''class made do organice all cuts.'''
	
	def __init__(self, line_catcher, pipeline):
		'''inits all works'''
			
		self.c, self.l, self.r = line_catcher.show()
		self.xy_liner('c')
		self.xy_liner('l')
		self.xy_liner('r')
		
	def xy_liner(self, dat):
		'''change list of y to xy list'''
		
		tmp = [(x,y) for x,y in enumerate(getattr(self,dat))]
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
	
	def __init__(self, line, pipeline):
		'''init'''
		
		self.line = line
		self.modyficator()
		
	def modyficator(self):
		'''make modifications'''
		
		pass
		
	def show(self):
		'''shows line'''
		
class xy_liner(overliner):
	'''changes data to xy format'''
	
	def modyficator(self):
		'''changes line''' 
				
class play_film(object):
	'''plays film'''
	
	def __init__(self, frame_source, pipeline):
		'''inits frame'''
		
		cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('frame', 800,800)
		
		while pipeline.show_data('GO'):
			
			try:
				frame = frame_source.show()
				
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			except ValueError:
				print('nmf')
				break
		b.end()			
	
class save_film(object):
	'''class to save film'''
	
	def __init__(self, overframe_like_class, pipeline, frame_source, save_file):
		'''saves frame by frame to file'''
		
		
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		w = pipeline.show_data('width')
		h = pipeline.show_data('height')
		out = cv2.VideoWriter(save_file,fourcc,30.0,(w,h))
		i = 0
		
		while frame_source.cio() and pipeline.show_data('num_to_save')>i:
			try:
				frame = frame_source.show()
				#print(i, frame.shape)
				out.write(frame)
				i+=1
			except ValueError:
				break
			
		out.release()
			
class pipeline(object):
	'''class to save data for modificators etc.'''
	
	def __init__(self):
		'''init start all fields'''
		
		#bisektor
		self.left = 250
		self.right = 500
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
		self.source = 'greentest.avi' #'greentest.avi' #'redtest1.avi' #'None' #None is for camera. Or use filepath
		self.num_to_save = 500
		#frame shape
		self.width = 0
		self.height = 0
		
	def set_data(self, **kwargs):
		'''sets data in pipeline'''
		
		for k,v in kwargs.items():
			setattr(self, k,v)
	
	def set_frame(self):
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

if __name__=='__main__':



#sprawdzic linechether
#napisać plotter
#napisać phaser
#napisać fuzzy phaser
#napisać fuzy phaser2

	c = pipeline()
	lom = [bright_frame,bis_frame]
	c.lom = lom
	print('pipeline ok')
	b = filmin(c)
	print('filmin ok')	
	#d = mod_clip(b,c)
	#d = bright_frame(b.show(),c)
	#e = bis_frame(b.show(),c)
	
	play_film(b,c)
	
	
	
	
	
