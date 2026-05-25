import jax
import jax.numpy as jnp

from ..core.protocols import LikelihoodModel
from typing import Optional

class MultivariateGuassianPDF(LikelihoodModel):
	"""
	Stateless probability density function. Calculates probabilities given inpust.
	
	Methods:
		evaluate() - Calculates the probability given an inpu
	"""
	
	def log_evaluate(self,
			  measurement:jax.Array,
			  predicted_measurement:jax.Array,
			  covariance:jax.Array,
			  predicted_state:Optional[jax.Array] = None
			  ) -> jax.Array:
		"""
		Evaluates a probable state given a measurement, predicted_measurement, and covariance
		using the multivariage guassian probability density function

		Params:
			measurement 				- jax.Array, sensor readings, measurements
			predicted_measurement 		- jax.Array, predicted measurement readings, sensor readings
			covariance					- jax.Array, covariance of the noise in the sensor readings, measurements

		Returns:
			p_yx						- jax.Array, probable state given the inputs
		"""
		
		k = measurement.shape[0]

		# Finding mahalanobis distance
		error = measurement -  predicted_measurement
		cov_inv = jax.scipy.linalg.solve(covariance, error, assume_a = 'pos')
		mahalanobis_distance = jnp.dot(error, cov_inv)

		# finding log probabilities
		sign, logdet = jnp.linalg.slogdet(covariance)
		log_prob = -0.5 *(k*jnp.log(2*jnp.pi) + logdet + mahalanobis_distance)
		return log_prob