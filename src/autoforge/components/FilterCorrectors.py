import jax
import jax.numpy as jnp

from functools import partial

from ..core.protocols import Corrector, LikelihoodModel, ObservationModel

# PARTICLE FILTER CORRECTOR
class PFCorrector(Corrector):
	"""
	Stateless Particle Filter Corrector used to generate
	weights on particles to correct prediction

	Methods:
		Public correct_step()			- Calculates weights on particle prediction at the next time step
		Public correct_trajectory()		- Calculates weights on particle predictions at each t in trajectory
	"""
	
	@partial(jax.jit, static_argnums=(0, 3, 5))
	def correct_step(
			self,
			weights: jax.Array, 
			particles: jax.Array,
			likelihood: LikelihoodModel,
			measurements: jax.Array,
			observation_model: ObservationModel,
			covariance: jax.Array
			) -> jax.Array:
		""" Calculates weights to correct prediction 
		Params:
			weights				- jax.Array, weighting vector for particles
			particles			- jax.Array (N, state_dim), prediction particles
			likelihood			- function, probability distribution function for weights
			measurements		- jax.Array, measurements of system behavior
			observation_model	- function, modeling of observations w.r.t. particles
			covariance			- jax.Arrya, matrix description of covariance 
		Returns:
			weights_next			- jax.Array, weighting vector for particle
		"""
		predicted_measurements = jax.vmap(observation_model.observe)(particles)
		
		# Batched log-sum-exp
		mapped_log_likelihood = jax.vmap(likelihood.log_evaluate, in_axes=(None, 0, None))(measurements, predicted_measurements, covariance) # mapping over particles
		log_w = jnp.log(weights + 1e-15) + mapped_log_likelihood
		log_w = log_w - jax.scipy.special.logsumexp(log_w)

		# Evaluating weights
		weights_next = jnp.exp(log_w)
		return weights_next
		
	@partial(jax.jit, static_argnums=(0, 2, 4))
	def correct_trajectory(self, 
						weights: jax.Array, 
						particles: jax.Array,
						likelihood: LikelihoodModel,
						measurement_sequence: jax.Array,
						observation_model: ObservationModel,
						covariance: jax.Array
						) -> jax.Array:
		""" 
		Calculates weights to correct prediction across a trajectory
		Params:
			weights					- jax.Array, weighting vector for particles
			particles				- jax.Array (N, state_dim), prediction particles
			likelihood				- function, probability distribution function for weights
			measurement_sequence	- jax.Array, measurements of system behavior
			observation_model		- function, modeling of observations w.r.t. particles
			covariance				- jax.Arrya, matrix description of covariance 
		Returns:
			weights_next			- jax.Array, weighting vector for particle
		"""

		# Use jax.lax.scan
		def scan_body(carry_weights, x):
			particles_t, measurement_t = x
			next_weights = self.correct_step(
										weights = carry_weights,
										particles = particles_t,
									  	likelihood = likelihood,
									  	measurements = measurement_t,
									  	observation_model = observation_model,
									  	covariance = covariance,
									  	)
			return next_weights, next_weights
	
		scan_inputs = (particles, measurement_sequence)

		final_weights, weight_trajectory = jax.lax.scan(f=scan_body, init=weights, xs=scan_inputs)

		return weight_trajectory