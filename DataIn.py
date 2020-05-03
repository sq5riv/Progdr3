import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

class filmin(object):
	'''Klasa pobieraków danych'''
	
	def __init__(self, source = 'None'):
		'''inits data'''
		
		if source == 'None':
			self.cap = cv2.VideoCapture()
		else: 
			self.cap = cv2.VideoCapture(source)
		
		cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('frame', 800,800)
	
	def cio(self):
		'''returns info about cap.isOpened'''
		
		return self.cap.isOpened()
		
	def end(self):
		'''finish work'''
		
		self.cap.release()
		cv2.destroyAllWindows()
		
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
		'''checks for color of frame and brigtest point'''
		
		if self.pipeline.show_data('color') == 'no':
			self.what_color()
		
			
	def modyficator(self):
		'''makes one color brights than others'''
		
		if self.pipeline.show_data('bright_on') == 'on':
			mul = self.pipeline.show_data('brightner')
			self.frame = self.frame * 255.0/float(mul)*1.5
		
	def what_color(self):
		'''finds color of laser'''
				
		val = {}	
		val3 = ''
		adr_max = np.argmax(self.frame)
		r = adr_max % 3
		if r == 1: val3 = 'green'
		elif r == 2: val3 = 'red'
		elif r == 0: val3 = 'blue'		
		val.update({'color':val3})
		val_max = np.amax(self.frame)
		bright = {'brightner':val_max}
		val.update(bright)
		self.pipeline.set_data(**val)

class bis_frame(overframe):
	
	def check(self):
		'''checks left and right bisector value'''
		
		self.left = self.pipeline.show_data('left')		
		self.right = self.pipeline.show_data('right')
	
	def horizontal_adder(self, y):
		'''ads horizontal line to frame'''
        
		np.put(self.frame, [range(768*y*3,768*(y+1)*3)], 255)
		
	def modyficator(self):
		'''adds bisector'''
		
		#print('inmod')
		self.horizontal_adder(288)
		self.vert_adder(self.left)
		self.vert_adder(self.right)
		center = (self.left + self.right)/2
		dev = abs(self.left - self.right)/2
		self.vert_short_adder(center, 30)
		self.vert_short_adder(self.right+dev, 30)
		self.vert_short_adder(self.left-dev,30)

	def vert_adder(self, x):
		'''asd vertical line to frame'''
		
		np.put(self.frame, [range(x*3, 576*767*3,768*3)], 255)
		np.put(self.frame, [range(x*3+1, 576*767*3,768*3)], 255)
		np.put(self.frame, [range(x*3+2, 576*767*3,768*3)], 255)

        
	def vert_short_adder(self, x, y):
		'''ads short vertical line fo frame'''

		np.put(self.frame, [range(int(x*3+768*3*(288-y)),int(x*3+768*3*(288+y)),768*3)], 255)
		np.put(self.frame, [range(int(x*3+1+768*3*(288-y)),int(x*3+768*3*(288+y)),768*3)], 255)
		np.put(self.frame, [range(int(x*3+2+768*3*(288-y)),int(x*3+768*3*(288+y)),768*3)], 255)
 		
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
		self.c_line = taker(self.center)
		self.l_line = taker(self.left)
		self.r_line = taker(self.right)
		
		plt.plot(self.c_line)
		plt.show()
		
	def show(self):
		'''shows data'''
		
		return (self.c_line, self.l_line, self.r_line)
		
	def taker(self, x):
		'''takes line from frame'''
		
		return np.take(self.frame, range(self.x*3+self.color,self.x*3+768*574*3+self.color,768*3))
			
		'''for i in range(768):
			self.c_line = np.take(self.frame, range(i*3+self.color, i*3+768*574*3+self.color, 768*3))
			plt.plot(self.c_line)
			plt.show()'''

class mod_clip(overframe):
	'''plays film'''
	
	def check(self):
		'''checks for list of modifications'''
		
		self.lom = self.pipeline.show_data('lom')
	
	
	def modifier(self):
		'''makes all modifies'''
			
		self.loo = []
		p = filmin(source)
		self.loo.append(val[0](p.show(), pipeline))
		for num in range(len(self.lom)-1):
			self.loo.append(self.lom[num+1](self.loo[-1].show(),pipeline))
	
	def show(self):
		'''shows modified frame'''
		
		return self.loo[-1].show()	
		
class play_film(object):
	'''plays film'''
	
	def __init__(self, overframe_like_class, pipeline, frame_source):
		'''inits frame'''
		
		cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('frame', 800,800)
		
		while frame_source.cio() and pipeline.show_data():
			
			try:
				d = overframe_like_class(frame_source.show(), pipeline)
				frame = d.show()
				#print(frame.shape)
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			except:
				print('nmf')
				break
		b.end()			
	
class save_film(object):
	'''class to save film'''
	
	def __init__(self, overframe_like_class, pipeline, frame_source, save_file):
		'''saves frame by frame to file'''
		
		
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(save_file,fourcc,30.0,(768,576))
		
		while frame_source.cio() and pipeline.show_data():
			try:
				d = overframe_like_class(frame_source.show(), pipeline)
				frame = d.show()
				out.write(frame)
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
		#list of modifications
		self.lom = []
		#info for live actions:
		self.GO = True
		
	def set_data(self, **kwargs):
		'''sets data in pipeline'''
		
		for k,v in kwargs.items():
			setattr(self, k,v)
				
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

	b = filmin('greentest.avi')
	c = pipeline()
	#play_film(bis_frame, c, b)
	save_film(bis_frame,c,b,'ende.avi')
	#v = [bis_frame]
	#cap = cv2.VideoCapture('greentest.avi')
	#mod_clip('greentest.avi', filmin, b2, *v)
	#print(b2.show_data('color'))
	#c = bright_frame(b.give_frame(),b2)
	
	
	
	

	#print(b2.brightner)
	#cv2.imwrite('frame.jpg',d.show())
	#time.sleep(30)
