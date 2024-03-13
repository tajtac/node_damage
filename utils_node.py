from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, vmap, jit, jacrev
from functools import partial
import jax.random as random
#from jax.experimental import optimizers
from jax.experimental.ode import odeint
import jax.example_libraries.optimizers as optimizers
from jax.scipy.optimize import minimize
from jax.lax import scan
from jax.nn import softplus
from jax.flatten_util import ravel_pytree

from jax.experimental.host_callback import id_print
rng = random.PRNGKey(2022)
import scipy



def init_layers(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    b = np.zeros(layers[i + 1])
    return Ws, b

def init_layers_nobias(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws

def init_params(layers, key):
    params_I1 = init_layers_nobias(layers, key)
    key, subkey = random.split(key)
    params_I2 = init_layers_nobias(layers, key)
    key, subkey = random.split(key)
    params_v = init_layers_nobias(layers, key)
    key, subkey = random.split(key)
    params_w = init_layers_nobias(layers, key)
    key, subkey = random.split(key)
    theta_v = 0.0
    theta_w = 1.57
    Psi1_bias = -5.0
    Psi2_bias = -5.0
    return ((params_I1, Psi1_bias), (params_I2, Psi2_bias), (params_v, theta_v), (params_w, theta_w))

def init_params_damage(key, Psi_layers=[1,3,3,1], f_layers=[1,3,3,1], G_layers=[1,3,3,1]): # For the time being use the same NN architecture for all
    params_Psi_list = init_params(Psi_layers, key) # Parameters of the Psi^o 's
    key, subkey = random.split(key)
    params_f_list = [init_layers_nobias(f_layers, key) for _ in range(4)] # Parameters of f functions
    key, subkey = random.split(key)
    params_G_list = [init_layers(G_layers, key) for _ in range(4)] # Parameters of G functions
    r_init = [0.0,0.0,0.0,0.0]
    params = [params_Psi_list, params_f_list, params_G_list]
    return params

def init_params_damage_simple(key, Psi_layers=[1,3,3,1], G_layers=[1,3,3,1]): # For the time being use the same NN architecture for all
    params_Psi = init_params(Psi_layers, key)[0]
    key, subkey = random.split(key)
    params_G = init_layers(G_layers, key)
    params = [params_Psi, params_G]
    return params

@jit
def forward_pass(H, params):
    Ws, b = params
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1]) + jnp.exp(b) #We want a positive bias
    return Y

@jit
def forward_pass_nobias(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1])
    return Y


# #NODE forward pass
# @jit
# def NODE(y0, params, steps = 200):
#     t0 = 0.0
#     dt = 1.0/steps
#     body_func = lambda y,t: (y + forward_pass_nobias(jnp.array([y]), params)[0]*dt, None)
#     out, _ = scan(body_func, y0, jnp.linspace(0,1,steps), length = steps)
#     return out
# NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

@jit
def RK_forward_pass(Y0, params):
  n = 4
  dt = 1.0/n
  def RK_step(Y,t):
    Y = jnp.array([Y])
    k1 = forward_pass(Y               , params)
    k2 = forward_pass(Y + 0.5*k1*dt   , params)
    k3 = forward_pass(Y + 0.5*k2*dt   , params)
    k4 = forward_pass(Y + k3*dt       , params)
    Y = Y + 1/6*dt*(k1 + 2*k2 + 2*k3 + k4)
    return (Y[0], None)
  out, _ = scan(RK_step, Y0, jnp.linspace(0,1,n), length = n)
  return out
RK_vmap = vmap(RK_forward_pass, in_axes=(0, None), out_axes=0)

@jit
def RK_forward_pass_nobias(Y0, params):
  n = 4
  dt = 1.0/n
  def RK_step(Y,t):
    Y = jnp.array([Y])
    k1 = forward_pass_nobias(Y               , params)
    k2 = forward_pass_nobias(Y + 0.5*k1*dt   , params)
    k3 = forward_pass_nobias(Y + 0.5*k2*dt   , params)
    k4 = forward_pass_nobias(Y + k3*dt       , params)
    Y = Y + 1/6*dt*(k1 + 2*k2 + 2*k3 + k4)
    return (Y[0], None)
  out, _ = scan(RK_step, Y0, jnp.linspace(0,1,n), length = n)
  return out
RK_vmap_nobias = vmap(RK_forward_pass_nobias, in_axes=(0, None), out_axes=0)


class NODE_model_aniso(): #anisotropic

    def __init__(self, params):
        NODE_weights, self.theta, self.Psi1_bias, self.Psi2_bias = params
        self.params_I1, self.params_I2, self.params_v, self.params_w = NODE_weights
    
    def Psi1(self, I1, I2, Iv, Iw):
        I1 = I1-3.0
        Psi_1 = RK_forward_pass_nobias(I1, self.params_I1)
        return Psi_1 + jnp.exp(self.Psi1_bias)
    
    def Psi2(self, I1, I2, Iv, Iw):
        I2 = I2-3.0
        Psi_2 = RK_forward_pass_nobias(I2, self.params_I2)
        return Psi_2 + jnp.exp(self.Psi2_bias)
    
    def Psiv(self, I1, I2, Iv, Iw):
        Iv = Iv-1.0
        Psi_v = RK_forward_pass_nobias(Iv, self.params_v)
        Psi_v = jnp.maximum(Psi_v, 0.0)
        return Psi_v
    
    def Psiv(self, I1, I2, Iv, Iw):
        Iw = Iw-1.0
        Psi_w = RK_forward_pass_nobias(Iw, self.params_w)
        Psi_w = jnp.maximum(Psi_w, 0.0)
        return Psi_w

class GOH_model(): #anisotropic
    def __init__(self, params):
        self.params = params
    
    def Psi1(self, I1, I2, Iv, Iw):
        C10, k1, k2, kappa = self.params

        E = kappa*(I1-3.0) + (1-3*kappa)*(Iv-1.0)
        E = jnp.maximum(E, 0.0)
        Psi1 = C10 + k1*kappa*E*jnp.exp(k2*E**2)
        return Psi1
    
    def Psi2(self, I1, I2, Iv, Iw):
        return 0.0
    
    def Psiv(self, I1, I2, Iv, Iw):
        C10, k1, k2, kappa = self.params

        E = kappa*(I1-3.0) + (1-3*kappa)*(Iv-1.0)
        E = jnp.maximum(E, 0.0)
        Psiv = k1*(1-3*kappa)*E*jnp.exp(k2*E**2)
        return Psiv
    
    def Psiw(self, I1, I2, Iv, Iw):
        return 0.0
    

def eval_Cauchy(lmbx,lmby, model):
    lmbz = 1.0/(lmbx*lmby)
    F = jnp.array([[lmbx, 0, 0],
                   [0, lmby, 0],
                   [0, 0, lmbz]])
    C = F.T @ F
    C2 = C @ C
    Cinv = jnp.linalg.inv(C)
    theta = model.theta
    v0 = jnp.array([ jnp.cos(theta), jnp.sin(theta), 0])
    w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
    V0 = jnp.outer(v0, v0)
    W0 = jnp.outer(w0, w0)

    I1 = C[0,0] + C[1,1] + C[2,2]
    trC2 = C2[0,0] + C2[1,1] + C2[2,2]
    I2 = 0.5*(I1**2 - trC2)
    Iv = jnp.einsum('ij,ij',C,V0)
    Iw = jnp.einsum('ij,ij',C,W0)

    Psi1 = model.Psi1(I1, I2, Iv, Iw)
    Psi2 = model.Psi2(I1, I2, Iv, Iw)
    Psiv = model.Psiv(I1, I2, Iv, Iw)
    Psiw = model.Psiw(I1, I2, Iv, Iw)

    p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
    S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*(I1*jnp.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0

    sgm = F @ (S @ F.T)
    return sgm
eval_Cauchy_aniso_vmap = vmap(eval_Cauchy, in_axes=(0,0,None), out_axes = 0)