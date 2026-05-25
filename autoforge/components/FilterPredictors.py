import jax
import jax.numpy as jnp


from ..core.protocols import Predictor, DynamicsModel
from typing import Any
from functools import partial

class PFPredictor(Predictor):
	"""
	Stateless Particle Filter Predictor used to generate a 
	prediction of system motion through time.

	Methods:
		Public predict_step() 			- Predicts the state at the next step
		Public predict_trajectory()		- Predicts the states at each t in trajectory
	
	"""

	@partial(jax.jit, static_argnums=(0, 4, 6))
	def predict_step(self, 
				  state:jax.Array, 
				  control:jax.Array, 
				  dt:float,
				  dynamics: DynamicsModel,
				  key: jax.Array, 
				  noise_distribution: Any,
				  standard_dev: jax.Array,
				  ) -> jax.Array:
		""" 
		Estimate the next state one step (dt) forward through time. 

		Params:
			state				- jax.Array, (N, state_dim) current state of the system
			control 			- jax.Array, current control input for the system
			dt					- float, time step between current and predicted step
			dynamics			- dynamic model to implement with this filter
			key					- jax.Array, key to generate noise simulation
			noise_distribution	- Any, distribution of the noise (guassian, normal)
			standard_dev		- jax.Array, standard deviation to reduce the distribution by

		Returns:
			prediction			- jax.Array, (N, state_dim) simulated step N particles per state  
		"""
		mapped_evolve = jax.vmap(dynamics.evolve, in_axes = (0, None, None))
		keys = jax.random.split(key, state.shape[0])
		noise = jax.vmap(lambda k: noise_distribution(k, shape=(state.shape[1],)))(keys)
		prediction = mapped_evolve(state, control, dt) + standard_dev * noise
		return prediction

	@partial(jax.jit, static_argnums=(0, 4, 6))
	def predict_trajectory(self, 
						initial_state:jax.Array,  
						control_sequence:jax.Array, 
						dt:float, 
						dynamics:DynamicsModel,
						keys:jax.Array,
						noise_distribution: Any,
						standard_dev: jax.Array,
						) -> jax.Array:
		"""Estimate the next state at each dt forward through time.

		Params:
			initial_state       	- jax.Array, starting particle matrix of shape (N, 4)
			control_sequence       	- jax.Array, sequence of inputs over time of shape (num_steps, dim_control)
			dt                  	- float, fixed time step spacing
			dynamics            	- Dynamics model instance
			keys                	- jax.Array, unique PRNG keys per time step of shape (num_steps, 2)
			noise_distribution  	- Any, (e.g., jax.random.normal)
			standard_dev        	- jax.Array, (state_dim,) deviation array

		Returns:
			trajectory_prediction	- jax.Array, complete history tensor of shape (num_steps, N, state_dim)
		"""

		inputs_over_time = (control_sequence, keys)

		# Use jax.lax.scan
		def scan_body(current_particles, current_inputs):
			control, key = current_inputs
			
			next_particles = self.predict_step(
										state = current_particles,
									  	control = control,
									  	dt = dt,
									  	dynamics = dynamics,
									  	key = key,
									  	noise_distribution = noise_distribution,
									  	standard_dev = standard_dev)
			return next_particles, next_particles
	
		_ , predicted_trajectory = jax.lax.scan(
			f=scan_body,
			init=initial_state,
			xs=inputs_over_time
		)

		return predicted_trajectory
