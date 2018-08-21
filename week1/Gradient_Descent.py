import numpy as np 
import pandas as pd # To read dataset
import matplotlib.pyplot as plt # Plotting
import time
from IPython import display

dataset = pd.read_csv('ADRvsRating.csv')
data = np.array(dataset)

numInstances = data.shape[0]

def SSE(m,b,data):

	totalError = 0.0

	for i in range(numInstances):

		adr = data[i,0] # Row 'i' column 'ADR'
		rating = data[i,1] # Row 'i' column 'Rating'

		# The real rating
		currentTarget = rating

		# Predicted rating with our current fitting line
		# y = mx + b
		currentOutput = m*adr + b

		# Compute square error
		currentSquareError = (currentTarget - currentOutput)**2

		# Add it to the total error
		totalError += currentSquareError
	sse = totalError/numInstances

	return sse


def gradient_descent_step(m,b,data):

	N= numInstances
	m_grad = 0
	b_grad = 0

	for i in range(N):

		# Get current pair (x, y)
		x = data[i, 0]
		y = data[i, 1]

		# Partial derivative respect 'm'
		dm = - ((2/N) * x * (y - (m*x + b)))

		# Partial derivative respect 'b'
		db = - ((2/N) * (y - (m*x + b)))

		# Update gradient
		m_grad = m_grad + dm
		b_grad = b_grad + db

	# Set the new 'better' updated 'm' and 'b'
	m_updated = m - 0.0001*m_grad
	b_updated = b - 0.0001*b_grad
	'''
	Important note: The balue '0.0001' that multiplies the 'learning rate', 
	but is's a concept
	out of the scope of this chaleenge. For now, just leave that there 
	and think about it like a 'smoother' of the learn, to prevent overshooting,
	that is, an extremely fast and uncontroled learning.
	'''

	return m_updated, b_updated

def gradient_descent_n_steps(m_starting,b_starting,data,steps):
	# For doing it many times in an easy way ;)
	m = m_starting
	b = b_starting
	for i in range(steps):
		m,b = gradient_descent_step(m,b,data)
	return m, b

M_STARTING = 2
B_STARTING = 3
NUM_STEPS = 1000

m_best, b_best = gradient_descent_n_steps(M_STARTING,B_STARTING,data,NUM_STEPS)

m = m_best
b = b_best
x = data[:,0]

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title('ADR vs Rating (CS:GO)')
ax.scatter(x=x, y=data[:,1], label='Data')
plt.plot(x, m*x + b, color='red', label='BEST Fitting Line')
ax.set_xlabel('ADR')
ax.set_ylabel('Rating')
ax.legend(loc='best')

#plt.show()

from mpl_toolkits.mplot3d import Axes3D

def error(x,y):
	return SSE(x,y,data)

m = np.arange(1,2,0.01)
b = np.arange(-0.5,0.5,0.01)

fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121, projection='3d')
ax.view_init(elev=20.0, azim=115)

X, Y = np.meshgrid(m, b)

zs = np.array([error(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z,cmap='hot')

ax.set_title('Gradient Descent')
ax.set_xlabel('slope (m)')
ax.set_ylabel('y-intercept (b)')
ax.set_zlabel('Error')

# PLOT2
ax2 = fig.add_subplot(122, projection='3d')
ax2.view_init(elev=50.0, azim=150)

X, Y = np.meshgrid(m,b)

zs = np.array([error(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax2.plot_surface(X, Y, Z, cmap='hot')

ax2.set_title('Gradient Descent')
ax2.set_xlabel('slope (m)')
ax2.set_ylabel('y-intercept (b)')
ax2.set_zlabel('Error')

#plt.show()

def make_plots(fig,axes,m_list,b_list,m,b,data,step):

	# PLOT1
	ax = axes[1]
	ax.set_title('ADR vs Rating (CS:GO)')
	ax.set_xlim(0,160)
	ax.set_ylim(0, 250)
	ax.set_xlabel('ADR')
	ax.set_ylabel('Rating')

	ax.scatter(x=data[:,0], y=data[:,1], label='Data')
	ax.plot(data[:,0], m*data[:,0] + b, color='red', label='First Fitting Line (Step %d)' % step)

	ax.legend(loc='best')

	#PLOT2
	ax2 = axes[0]
	ax2.cla()

	ax2.set_title('Gradient Search')
	ax2.set_ylim(0.9, 1.5)
	ax2.set_ylim(0.999, 1.006)
	ax2.set_xlabel('slope (m)')
	ax2.set_ylabel('y y-intercept (b)')

	ax2.text(-1.15, 0.97, 'Iteration: '+str(step),
		verticalalignment='top',horizontalalignment='left',
		transform=ax.transAxes,
		color='dimgrey',fontsize=10)
	ax2.text(-1.15, 0.93, 'm = '+str(round(m,3)) + ', b = '+str(round(b,3)),
		verticalalignment='top', horizontalalignment='left',
		color='dimgrey', fontsize=10)

	ax2.plot(m_list, b_list, color='black', linewidth = 0.5)
	ax2.scatter(m,b,marker='^')

	fig.canvas.draw()

def gradient_descent_n_steps_with_plot(m_starting, b_starting, data, steps):
	# For doing it many times in an easy way ;)
	
	fig,axes = plt.subplots(1,2,figsize=(10,7))

	m_list = [m_starting]
	b_list = [b_starting]
	m = m_starting
	b = b_starting
	plt.cla()
	for i in range(steps):
		step = i
		make_plots(fig,axes,m_list,b_list,m,b,data,step)

		m, b = gradient_descent_step(m,b,data)
		m_list.append(m)
		b_list.append(b)

		time.sleep(10/steps)
		plt.cla()
		############
	make_plots(fig,axes,m_list,b_list,m,b,data,step)
#end def

def error_plot(fig,ax,error_list,error,data,step):
    #PLOT2

    ax.cla()
    
    ax.set_title('Error (step %d)' % step)
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Error')
    
    ax.plot(np.arange(0,len(error_list)),error_list)    
    fig.canvas.draw()
    
def gradient_descent_n_steps_with_error_plot(m_starting,b_starting,data,steps): #For doing it many times in an easy way ;)
    
    fig,ax = plt.subplots(1,1,figsize=(10,7))
    
    m = m_starting
    b = b_starting
    error_list = list()
    
    
    error = SSE(m,b,data)
    error_list.append(error)
    
    plt.cla()
    for i in range(steps):
        step = i
        error_plot(fig,ax,error_list,error,data,step)
        
        m,b = gradient_descent_step(m,b,data)
        error = SSE(m,b,data)
        error_list.append(error)
        
        time.sleep(10/steps)
        plt.cla()
        #############
            
    error_plot(fig,ax,error_list,error,data,step)
#end def

#RUN IT!

m = 1
b = 1
steps = 10
gradient_descent_n_steps_with_error_plot(m,b,data,steps)
