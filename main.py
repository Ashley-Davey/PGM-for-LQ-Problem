import numpy as np
from scipy.integrate import odeint
import time
from matplotlib import pyplot as plt, rcParams







rcParams['figure.dpi'] = 600

def mod(X):
    #returns op-norm for matrices, euclidean norm for vectors
    if len(X.shape) == 1:
        return np.sqrt(np.sum(X ** 2))
    else:
        return np.real(np.sqrt(np.max(np.linalg.eigvals(X.T @ X))))

def mod0(alpha, beta = None):
    #returns |phi|_0 if alpha and beta, else [phi]_1
    alpha0 = max(map(mod, alpha))
    if beta is not None:
        beta0  = max(map(mod, beta))
        return max(alpha0, beta0)
    else:
        return alpha0
    
def bat_mult(A, b):
    #computes Ab when b is a batch of vectors
    return np.squeeze(A @ np.expand_dims(b, -1), -1)
    
#Set up problem

class Algo(object):
    def __init__(self, n = 2, m = 2, T = 1.0, r = None, loud = True, plot = True, rand = False):
        
        #output config
        self.loud = loud
        self.plot = plot
                
        
        self.print = print if self.loud else (lambda x: None)
        self.run_plots = self.run_plots if self.plot else (lambda : None)
        
        #model parameters        
        self.n = n
        self.m = m
        self.T = T
        
        if rand:
            np.random.seed(1) #for replicability 
        
         
        self.A = (np.random.rand(n,n) - 0.5)  / (  2 * max(n, 5) ** 1.5)
        self.B = (np.random.rand(n,m) - 0.5)   / ( 2 *  max(n, 5) ** 1.5 )
        self.C = (np.random.rand(n,n) - 0.5)  / (  2 * max(n, 5)  ** 1.5 ) 
        self.D = (np.random.rand(n,m) - 0.5)  / (  2 * max(n, 5)  ** 1.5 )  
        
        
        Q_temp = (np.random.rand(n,n) - 0.5) 
        
        self.Q = Q_temp + Q_temp.T
        
        if r == None:
        
            mu = 0
            while mu < 0.4:
                
                self.R = (np.random.rand(m,m) - 0.5) / np.sqrt(self.m) + np.eye(m)
                mu     = np.real(np.min(np.linalg.eigvals(self.R.T @ self.R)))


        else:
            self.R = np.eye(m) * r

            
        #S is m by n
        self.S = np.random.rand(m,n) - 0.5 
        
        
        G_temp = (np.random.rand(n,n) - 0.5) / (2 )
        
        self.G = G_temp + G_temp.T


        self.gamma = (np.random.rand(n) - 0.5) 
        
        self.sigma = (np.random.rand(n) - 0.5) 
        self.q     = (np.random.rand(n) - 0.5) 
        
        self.p     = (np.random.rand(m) - 0.5) 
        self.g     = (np.random.rand(n) - 0.5)
        
        x1 = np.concatenate([self.Q, self.S.T], axis = 1)
        x2 = np.concatenate([self.S, self.R  ], axis = 1)
        
        x = np.concatenate([x1, x2], axis = 0)

        self.warn(np.all(np.linalg.eigvals(x) > 0), 'Running matrix positive definite')
        
        
        
        #config for scheme
        self.max_iter = 200
        self.display_steps = 5
        self.times = 100
        
        
        
    def run(self):
        self.start_time = time.time()
        self.solve().setup()
        self.setup_time = time.time() - self.start_time
        self.print(f'Setup finished at time {self.setup_time:.1f} seconds. Iterating')
        self.iterate()
        self.final_time = time.time() - self.start_time
        self.print(f'Run finished at time {self.final_time:.1f} seconds')
        self.run_plots()
        
        return self
        
    def solve(self):
        #solve explicitly
        
        def fun(t, a):
            a_temp = a.reshape(self.n, self.n)
            K = self.D.T @ (a_temp @ self.D) + self.R
            L = self.B.T @ a_temp + self.D.T @ (a_temp @ self.C) + self.S
    
            ret  = (
                a_temp @ self.A
                + self.A.T @ a_temp
                + self.C.T @ (a_temp @ self.C)
                + self.Q 
                - L.T @ (np.linalg.inv(K) @ L)
                )
            
            return -1 * ret.reshape(self.n ** 2)
        

        
        self.a_hat = odeint(
            fun,
            self.G.reshape(self.n ** 2),
            np.linspace(self.T, 0, self.times),
            tfirst = True
            ).reshape(self.times, self.n, self.n)[::-1]
        
        
        K = self.D.T @ (self.a_hat @ self.D) + self.R
        L = self.B.T @ self.a_hat + self.D.T @ (self.a_hat @ self.C) + self.S
        
        
        def fun(t, b):
            a = self.interp(self.a_hat, t)

            K_temp = self.interp(K, t)

            L_temp = self.interp(L, t)


            M = self.B.T @ b + self.D.T @ (a @ self.sigma) + self.p
            
            ret  = (
                self.A.T @ b
                + a @ self.gamma
                + self.C.T @ (a @ self.sigma)
                + self.q
                - L_temp.T @ (np.linalg.inv(K_temp) @ M)
                )
            
            return -1 * ret.reshape(self.n)
        

        
        self.b_hat = odeint(
            fun,
            self.g,
            np.linspace(self.T, 0, self.times),
            tfirst = True
            ).reshape(self.times, self.n)[::-1]        
        
        M1 = bat_mult(self.B.T, self.b_hat)
        M2 = bat_mult(self.D.T, self.a_hat @ self.sigma) + self.p
        
        M = M1 + M2
        
        
        def fun(t, xi):
            a = self.interp(self.a_hat, t)
            b = self.interp(self.b_hat, t)
            
            K_temp = self.interp(K, t)

            M_temp = self.interp(M, t)
            ret  = (
                b @ self.gamma
                + 0.5 * self.sigma @ (a @ self.sigma)
                - 0.5 * M_temp.T @ (np.linalg.inv(K_temp) @ M_temp)
                )
            
            return -1 * ret
        

        
        self.xi_hat = odeint(
            fun,
            0,
            np.linspace(self.T, 0, self.times),
            tfirst = True
            ).reshape(self.times)[::-1]          
        
        
        self.alpha_hat = - np.linalg.inv(K) @ L

        
        self.beta_hat  = -bat_mult(np.linalg.inv(K), M)
        
        
        self.soln = self.value(self.a_hat[0], self.b_hat[0], self.xi_hat[0])
                
        return self
        

    #helper functions
    def interp(self, vec, t):
        #interpolate for vec at t

        pre = max(min(int(t * self.times / self.T ), self.times - 1), 0)
        
        # return vec[pre]
        post = min(pre + 1, self.times - 1)
        
        c = t * self.times / self.T - pre
        
        
        return vec[pre] * c + vec[post] * (1 - c)
        
    
    def value(self, a, b, xi):
        
        #gets value, given intial matrix a
        x = np.ones(self.n)
        return 0.5 * (a @ x) @ x + b @ x + xi
    
    def warn(self, foo, err):
        if not foo:
            self.print('warning: assumption not true: ' + err)
            self.error = True
    
    def setup(self):
        self.print(f'Setting up problem with n = {self.n}, m = {self.m}')
        #initialise
        self.alpha0 = np.ones((self.times, self.m, self.n)) / (max(self.n, 5) * max(self.m, 5))
        self.beta0  = np.ones((self.times, self.m))  / max(self.m, 5)
        
        #calculate constants
        mu    = np.real(np.sqrt(np.min(np.linalg.eigvals(self.R.T @ self.R))))
        mod_R = np.real(np.sqrt(np.max(np.linalg.eigvals(self.R.T @ self.R))))
        self.tau = 0.9 * min(
                2 / (mu + mod_R),
                (mu + mod_R) / (2 * mu * mod_R)
                )
        
    
        
        self.error = False        

        self.warn( mu > 0, 'mu > 0')
        
        self.warn(
            self.tau < min(2 / (mu + mod_R), (mu + mod_R) / (2 * mu * mod_R)),
            'self.tau < min(2 / (mu + mod_R), (mu + mod_R) / (2 * mu * mod_R))'
            )
        
        temp = self.error
        self.error = False
        
        self.wont_converge = self.error
        
        self.error = (temp or self.error) 

        
        return self

    def iterate(self):

        try:

            times  = [0]
            values = [None]
            
            epoch = 1
            diff = 1
            alpha = self.alpha0
            beta  = self.beta0
            self.error0 = mod0(self.alpha0 - self.alpha_hat, self.beta0 - self.beta_hat)  / mod0(self.alpha_hat, self.beta_hat)

            time_now = time.time() - self.start_time
            
            self.print(f'Solution: {self.soln:.2e}')
            self.print(f'Initial control error: {self.error0:.2e}')
            self.print(f'Learning rate: {self.tau:.2e}')

            self.print('epoch \t cont err \t val err \t time')    
            self.print(f'{0:4d} \t {self.error0:.2e}  \t -------- \t {int(time_now):4d}')

            
            errors = {
                'control': [self.error0],
                'a': [None],
                'value': [None]
                }
            #iterate
            while epoch <= self.max_iter and (diff > 1e-3 if epoch > 10 else True):
                
                pre_error = mod0(alpha - self.alpha_hat, beta - self.beta_hat) / mod0(self.alpha_hat, self.beta_hat)

                
                #update a
                def fun_a(t, a):
                    a_temp = a.reshape(self.n, self.n)
                    alpha_temp = self.interp(alpha, t)
                    ret  = (
                        a_temp @ (self.A + self.B @ alpha_temp) 
                        + self.A.T @ a_temp
                        + self.C.T @ (a_temp @ (self.C + self.D @ alpha_temp))
                        + self.Q + self.S.T @ alpha_temp
                        )
                    
                    return -1 * ret.reshape(self.n ** 2)
                
                a = odeint(
                    fun_a,
                    self.G.reshape(self.n ** 2),
                    np.linspace(self.T, 0, self.times),
                    tfirst = True
                    ).reshape(self.times, self.n, self.n)[::-1]
                
                
            
                #update b
                def fun_b(t, b):
                    a_temp = self.interp(a, t)
                    b_temp = b
                    beta_temp = self.interp(beta, t)
                    ret  = (
                        self.A.T @ b_temp
                        + (a_temp @ self.B + self.C.T @ (a_temp @ self.D) + self.S.T) @ beta_temp
                        + a_temp @ self.gamma + self.C.T @ (a_temp @ self.sigma) + self.q
                        )
                    return -1 * ret
                
                b = odeint(
                    fun_b,
                    self.g,
                    np.linspace(self.T, 0, self.times),
                    tfirst = True
                    ).reshape(self.times, self.n)[::-1]
                
                




            
            
                #update alpha
                
                alpha = (
                    alpha
                    - self.tau * (
                        self.B.T @ a
                        + self.D.T @ ( a @ (self.C + self.D @ alpha))
                        + self.R @ alpha
                        + self.S
                        )
                    )
                
                
                #update beta            
                    
                beta = (
                    beta
                    - self.tau * np.squeeze(
                        self.B.T @ np.expand_dims(b,-1)
                        + (
                            self.D.T @ (a @ self.D)
                            + self.R
                            ) @ np.expand_dims(beta, -1)
                        + self.D.T @ np.expand_dims(a @ self.sigma, -1)
                        + np.expand_dims(self.p, -1),
                        -1)
                    )
                
                                
                #calculate error
                
                error = mod0(alpha - self.alpha_hat, beta - self.beta_hat) / mod0(self.alpha_hat, self.beta_hat)
                
                diff = abs(pre_error - error) / abs(pre_error)
                error_a = mod0(a - self.a_hat, b - self.b_hat) / mod0(self.a_hat, self.b_hat)

                value = self.value(a[0], b[0], self.xi_hat[0])
                
                time_now = time.time() - self.start_time
                rel_err = np.abs(value - self.soln) / np.abs(self.soln)
                
                times.append(time_now)
                errors['control'].append(error)
                errors['a'].append(error_a)
                errors['value'] .append(rel_err)
                values.append(value)

                
                if epoch %  self.display_steps == 0:
                    self.print(f'{epoch:4d} \t {error:.2e} \t {rel_err:.2e} \t {int(time_now):4d}')
                epoch += 1
                    
                        
        except KeyboardInterrupt:
            self.print(f'Manually terminated at epoch {epoch + 1}' )
            
        
        if diff < 1e-3:
            self.print(f'Suspected plateau, terminated at epoch {epoch + 1}')
        elif epoch == self.max_iter:
            self.print(f'Reached max iteration, terminated at epoch {epoch + 1}')
        
        
        #print final error
        
        
        errorf = mod0(alpha - self.alpha_hat, beta - self.beta_hat)
        
        self.print(f'Final control error: {errorf:.5e}')
        

        approx = self.value(a[0], b[0], self.xi_hat[0])
        err = np.abs(approx - self.soln) / self.soln
        

        self.print(f'Solution: {self.soln:.5e}, Approx: {approx:.5e}, rel err: {err:.2e} ({100 * err:.2f}%)')
        
        self.results = {
            'errors': errors,
            'times': times,
            'values': values,
            'epochs': np.arange(len(times))
            }
        

        
        
      
    def run_plots(self):
        
        times  = self.results['times']
        errors = self.results['errors']
        values = self.results['values']
        
        epochs = np.arange(len(times))
        
        #plot control error
        
        plt.figure()
        plt.grid(True, alpha = 0.5)
        plt.plot(epochs, errors['control'], label = 'error')
        plt.legend()
        plt.xlabel('Epoch ($k$)')
        plt.ylabel('Relative Control Error')
        
        
        plt.figure()        
        plt.grid(True, alpha = 0.5)
        plt.plot(epochs, errors['control'], label = 'error')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Epoch ($k$)')
        plt.ylabel('Relative Control Error')
        
        #plot matrix error
        
        
        plt.figure()        
        plt.grid(True, alpha = 0.5)
        plt.plot(epochs, errors['a'], label = 'error')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('epoch ($k$)')
        plt.ylabel('Relative Error of $a$')
        

        #plot value error
        
        
        plt.figure()
        plt.grid(True, alpha = 0.5)
        plt.plot(epochs, values, label = 'value')
        plt.axhline(self.soln, color  = 'r', label = 'solution', linestyle = 'dotted')
        plt.ylim(self.soln * (1 - 1e-1), self.soln * (1 + 1e-1))
        plt.legend()
        plt.xlabel('Epoch ($k$)')
        plt.ylabel('Value')
        
        
        return self
        
    
if __name__ == '__main__':    
    Algo(n = 100, m = 100).run()
    

