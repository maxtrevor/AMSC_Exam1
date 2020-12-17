import numpy as np
import time
import matplotlib.pyplot as plt

max_iter = 5000
convergence_tol = 10**-10
xmin = np.array([0.861289,0.373492])


def StochGradDescent(fun, grad, dat, x0, stepMethod, bsz, step_const):
    t0 = time.time()
    n = len(dat)
    bsz = min(n, bsz)
    
        
    x_rec = np.zeros((max_iter+1,2))
    f_rec = np.zeros(max_iter+1)
    norm_grad_rec = np.zeros(max_iter+1)
    f_rec[0]=fun(x0, dat)
    x_rec[0]=x0
    
    
    ts = np.zeros(max_iter)
        
    for k in range(max_iter):
        if bsz<n:
            inds = np.random.permutation(np.arange(n))[:bsz]
            subdat=dat[inds]
        else: subdat = dat
        g = grad(x_rec[k], subdat)
        norm_grad_rec[k] = np.linalg.norm(g)
        a = stepsize(x_rec[k], -g, g, fun, step_const, stepMethod, k, subdat)        
        x_rec[k+1] = x_rec[k]-a*g  
        f_rec[k+1] = fun(x_rec[k+1],dat)
        
        
        
        if k>=10:
            m = np.max(norm_grad_rec[k-10:k])
            if m<convergence_tol:
                print('failed, methd = '+stepMethod)
                break
        
        ts[k] = time.time()-t0
    
    norm_grad_rec[k+1] = np.linalg.norm(grad(x_rec[k+1], subdat))    
    f_rec[k+2:]=f_rec[k+1]
    x_rec[k+2:]=x_rec[k+1]
    norm_grad_rec[k+2:]=norm_grad_rec[k+1]
    x= x_rec[k+1]
    
    return (x, x_rec, f_rec, norm_grad_rec, k, ts)

def stepsize(x, d, g, fun, const, stepMethod, k, dat ):
    if stepMethod=='constant':
        return const
    if stepMethod =='log':
        if k<=300: return const
        elif np.log(k-300)<1/const: return const
        else: return 1/np.log(k-300)
    if stepMethod=='harmonic':
        if k==0: return 1
        else: return 1/k
      
    
def fun(x,dat):
    aux = (np.maximum(x[0]*dat-x[1],0)+np.cos(dat)-1)**2
    return np.sum(aux)/12

def grad(x,dat):
    aux = np.maximum(x[0]*dat-x[1],0)+np.cos(dat)-1
    dfda = np.sum(dat*np.heaviside(x[0]*dat-x[1],0)*aux)/6
    dfdb = -1*np.sum(np.heaviside(x[0]*dat-x[1],0)*aux)/6
    return np.array([dfda,dfdb])

dat = np.linspace(0,np.pi/2,6)
x0 = np.array([1,0])

# question 2

#g = grad(x0,dat)
#alpha = 1/(g[0]-2*g[1]/np.pi)
#print(alpha)
#
#step_const = 0.99*alpha
#(x, x_rec, f_rec, norm_grad_rec, k, ts) = StochGradDescent(fun, grad, dat, x0, 'constant', 6, step_const)
#print(x)
#print(k)
#print(f_rec[k+1])
#
#step_const = 1.3
#(x2, x_rec2, f_rec2, norm_grad_rec2, k2, ts2) = StochGradDescent(fun, grad, dat, x0, 'constant', 6, step_const)
#print(x2)
#print(k2)
#print(f_rec2[k2+1])
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(f_rec,label=r'$\alpha = 1.49$')
#ax.plot(f_rec2,label=r'$\alpha = 1.31$')
#ax.set_yscale('log')
#ax.set_ylabel('Function Value')
#ax.set_xlabel('Iteration')
#ax.legend()
#plt.show()



# question 3
t0 = time.time()
runs = 500

f_recs1 = np.zeros((runs,max_iter+1))
f_recs2 = np.zeros((runs,max_iter+1))
f_recs3 = np.zeros((runs,max_iter+1))
x_recs1 = np.zeros((runs,max_iter+1,2))
x_recs2 = np.zeros((runs,max_iter+1,2))
x_recs3 = np.zeros((runs,max_iter+1,2))
for i in range(runs):    
    (x, x_recs1[i], f_recs1[i], norm_grad_rec, k, ts) = StochGradDescent(fun, grad, dat, x0, 'log', 1, 1.31)
    (x2, x_recs2[i], f_recs2[i], norm_grad_rec2, k2, ts2) = StochGradDescent(fun, grad, dat, x0, 'constant', 1, 1.31)
    (x3, x_recs3[i], f_recs3[i], norm_grad_rec3, k3, ts3) = StochGradDescent(fun, grad, dat, x0, 'harmonic', 1, 1.31)

f_rec1 = np.mean(f_recs1,axis=0)
f_rec2 = np.mean(f_recs2,axis=0)
f_rec3 = np.mean(f_recs3,axis=0)

x_recs1 = x_recs1-xmin
x_recs2 = x_recs2-xmin
x_recs3 = x_recs3-xmin
dists1 = np.linalg.norm(x_recs1,axis=2)
dists2 = np.linalg.norm(x_recs2,axis=2)
dists3 = np.linalg.norm(x_recs3,axis=2)
d1 = np.mean(dists1,axis=0)
d2 = np.mean(dists2,axis=0)
d3 = np.mean(dists3,axis=0)
print(d2.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(f_rec2,label='Constant')
ax.plot(f_rec1,label='Logarithmic')
ax.plot(f_rec3,label='Harmonic')
ax.set_yscale('log')
ax.set_ylabel('Function Value')
ax.set_xlabel('Iteration')
ax.legend()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(d2,label='Constant')
ax.plot(d1,label='Logarithmic')
ax.plot(d3,label='Harmonic')
ax.set_yscale('log')
ax.set_ylabel('Distance from minimum')
ax.set_xlabel('Iteration')
ax.legend()
plt.show()

print(time.time()-t0)
