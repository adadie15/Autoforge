import jax
import jax.numpy as jnp

from ..core.protocols import StateEstimator, Predictor, Corrector, LikelihoodModel, ObservationModel, DynamicsModel

class ParticleFilter(StateEstimator):
	"""
	Forces typehinting and abstraction for fitler class (e.g. EKF, Particle Filter) 
	stateful (should keep internal state for covariance calculation)
	
	Fields:
		Private _predictor			- Predictor, stateless prediction engine
		Private _corrector			- Corrector, stateless correction engine
		Private _weights			- jax.Array, weight values for particles
		Private _particles			- jax.Array, array of particle predictions
		Private _n_particles		- int, number of particles
		Private _standard_dev		- float, standard deviation for dynamics evolution
		Private _covariance			- jax.Array, matrix of covariances for observation model
		Private _observation_model	- ObservationModel, observation model used for correction
		Private _dynamics			- DynamicsModel, dynamics model used for predictions
		Private _sensor_positions	- jax.Array, sensor positiongs RESERVED FOR FUTURE USE
		Private _key				- jax.Array, randomization key for jax
		Private _Neff_threshold		- float, threshold resample number to prevent weight collapse
	Methods:
		Public sequential_importance_resampling()
		Public estimate()
	
	"""

	def __init__(self, 
			  predictor:Predictor,
			  key:jax.Array, 
			  corrector:Corrector,
			  likelihood:LikelihoodModel,
			  init_mean: jax.Array,
			  init_cov: jax.Array, 
			  n_particles:int, 
			  standard_dev:float, 
			  measurement_cov:jax.Array,
			  Neff_threshold:float, 
			  observation_model:ObservationModel, 
			  dynamics:DynamicsModel
			  ) -> None:
		
		# initialize particles from Gaussian prior
		self._init_mean = init_mean
		self._predictor = predictor
		self._corrector = corrector
		self._likelihood = likelihood
		self._init_cov = init_cov
		self._standard_dev = standard_dev
		self._n_particles = n_particles
		self._measurement_cov = measurement_cov
		self._observation_model = observation_model
		self._dynamics = dynamics
		self._Neff_threshold = Neff_threshold

		# generate n_particles based on key
		self._key = key
		key, subkey = jax.random.split(key)
		self._particles = jax.random.multivariate_normal(subkey, 
											mean=self._init_mean, 
											cov=self._init_cov, 
											shape=(n_particles,))
		
		# generate initial weights
		self._weights = jnp.ones(n_particles)/n_particles

		# Save updated key
		self._key = key

	def _compute_Neff(self) -> None:
		"""
		calculates the effective number of samples 
		Returns
			Result		-float, 1/sum(w^2)
		"""
		result = 1.0 / jnp.sum(self._weights **2)
		return result


	def _resample(self) -> None:
		"""
		resamples the weights to prevent weight collapse
		"""
		self._key, r_key = jax.random.split(self._key)

		indices = jax.random.choice(key=r_key, a=self._n_particles, shape=(self._n_particles,), p=self._weights)

		self._particles = self._particles[indices]

		self._weights = jnp.ones(self._n_particles)/self._n_particles


		

	def estimate(self, control, measurement, dt) -> jax.Array:
		""" predicts and corrects for next time step"""
		self._key, p_key = jax.random.split(self._key)

		self._particles = self._predictor.predict_step(
												state=self._particles,
												control=control,
												dt=dt,
												dynamics=self._dynamics,
												key=p_key,
												noise_distribution=jax.random.normal,
												standard_dev=self._standard_dev
												)
		
		self._weights = self._corrector.correct_step(
											weights = self._weights, 
											particles = self._particles,
											likelihood = self._likelihood,
											measurements = measurement,
											observation_model = self._observation_model,
											covariance=self._measurement_cov
											)
		
		result = jnp.sum(self._weights[:, None] * self._particles, axis=0)

		neff = self._compute_Neff()

		if (neff < self._Neff_threshold):
			self._resample()
		
		return result
