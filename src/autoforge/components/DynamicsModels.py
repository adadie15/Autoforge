import jax
import jax.numpy as jnp

from ..core.protocols import DynamicsModel

class Unicycle(DynamicsModel):
	"""
	Defines Unicycle model
	Stateless model
	
	method:
		Public evolve()		- Steps the model forward in time
	"""
	
	@staticmethod
	@jax.jit
	def evolve(state: jax.Array, control: jax.Array, dt:float) -> jax.Array:
		"""
		Steps Unicycle Model forward through time.
		
		Params:
			state 		- jax.Array, [x, y, theta] current state of the system
			control		- jax.Array, [u, omega] current control input
		Returns:
			new_state	- jax.Array, next state
		"""
		x, y, theta = state
		u, omega = control
		
		x_new = 	x + 	dt*u*jnp.cos(theta)
		y_new = 	y + 	dt*u*jnp.sin(theta)
		theta_new = theta + dt*omega

		new_state = jnp.array([x_new, y_new, theta_new])
		return new_state
	

class ExtendedUnicycle(DynamicsModel):
	"""
	Defines an Extended Unicycle model
	Stateless model
	
	method:
		Public evolve()		- Steps the model forward in time
	"""
	
	@staticmethod
	@jax.jit
	def evolve(state: jax.Array, control: jax.Array, dt:float) -> jax.Array:
		"""
		Steps Extended Unicycle forward through time.
		
		Params:
			state 		- jax.Array, [x, y, theta, v] current state of the system
			control		- jax.Array, [omega, a] current control input
		Returns:
			new_state	- jax.Array, next state
		"""
		x, y, theta, v = state
		omega, a = control
		
		x_new = 	x + 	dt*v*jnp.cos(theta)
		y_new = 	y + 	dt*v*jnp.sin(theta)
		theta_new = theta + dt*omega
		v_new = v + dt*a

		new_state = jnp.array([x_new, y_new, theta_new, v_new])
		return new_state