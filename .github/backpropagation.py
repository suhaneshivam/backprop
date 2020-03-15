import numpy as np
class NueralNetwork:
	def __init__(self,layers,alpha=0.1):
		self.w=[]
		self.layers=layers
		self.alpha=alpha
		
		for i in np.arange(0,len(layers)-2):
			w=np.random.randn(layers[i]+1,layers[i+1]+1)
			self.w.append(w/np.sqrt(layers[i]))
			
		w=np.random.rand(layers[-2]+1,layers[-1])
		self.w.append(w/np.sqrt(layers[-2]))
	def __repr__(self):
		return "NueralNetwork: {}".format("-".join(str(l) for l in self.layers))
	
	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))
	
	def sigmoid_deriv(self,x):
		return x*(1-x)
	
	def fit(self,x,y,epochs=1000,displayUpdate=100):
		x=np.c_[x,np.ones(x.shape[0])]
		
		for epoch in np.arange(0,epochs):
			
			for (X,target) in zip(x,y):
				self.fit_partial(X,target)
			
			if epoch==0 or (epoch+1)%displayUpdate==0:
				loss=self.calculate_loss(x,y)
				print("[INFO]: epoch={},loss={}".format(epoch+1,loss))
				
	def fit_partial(self,x,y):
		A=[np.atleast_2d(x)]
		for layer in np.arange(0,len(self.w)):
			net=A[layer].dot(self.w[layer])
			out=self.sigmoid(net)
			
			A.append(out)
		
		error=A[-1] -y
		D=[error*self.sigmoid_deriv(A[-1])]
		
		for layer in np.arange(len(A)-2,0,-1):
			delta=D[-1].dot(self.w[layer].T)
			delta=delta*self.sigmoid_deriv(A[layer])
			D.append(delta)
		
		D=D[::-1]
		for layer in np.arange(0,len(self.w)):
			self.w[layer] += -self.alpha * A[layer].T.dot(D[layer])
	def predict(self,x,addBias=True):
		p=np.atleast_2d(x)
		
		if addBias:
			p=np.c_[p,np.ones(p.shape[0])]
		
			
		for layer in np.arange(0,len(self.w)):
			p=self.sigmoid(np.dot(p,self.w[layer]))
		return p
	def calculate_loss(self,x,targets):
		targets=np.atleast_2d(targets)
		predictions=self.predict(x,addBias=False)
		loss=0.5*np.sum((predictions-targets)**2)
		return loss
		
			
			
			
				
		
		
				
				
		
   	
		
	   
				
	
	
