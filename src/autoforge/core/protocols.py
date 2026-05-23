from typing import Protocol, runtime_checkable
import jax

# Class List: Barrier Functions
@runtime_checkable # remove for C++
class BarrierGeometry(Protocol):
	"""Forces typehinting and barrier geometry type for DECBF object. 
	NOTE: Barrier functions are ONLY well-defined for smooth, continuous functions
	but may be nonlinear.
	"""

	def get_h(self, p_agent:jax.Array) -> jax.Array:
		""" returns geometric expression for barrier eval"""
		...
	
	def get_center(self) -> jax.Array:
		""" returns center position of the geometry"""
		...

	def set_center(self, center:jax.Array) -> None:
		""" sets the center position of the geometry"""
		...

	def copy(self) -> 'BarrierGeometry':
		""" returns a memory safe copy of the geometry"""
		...

# Class List: Estimator Protocols
@runtime_checkable # remove for C++
class Predictor(Protocol):
	"""
	Forces abstraction of predictor classes for state estimation
	"""
	def predict_step(self, state:jax.Array, control:jax.Array, dt:float) -> jax.Array:
		""" Predicts the next time step (online processing, Realtime)"""
		...
	
	def predict_trajectory(self, initial_state:jax.Array, states:jax.Array, control:jax.Array, dts:jax.Array) -> jax.Array:
		"""Predictions on full trajectory (offline processing)"""
		...

@runtime_checkable # remove for C++
class Corrector(Protocol):
	"""
	Forces abstraction of corrector classes for state estimation
	"""
	def correct_step(self, predicted_state: jax.Array, measurement:jax.Array) -> jax.Array:
		""" Corrects prediction of the next time step (online processing, Realtime)"""
		...

	def correct_trajectory(self, predicted_states:jax.Array, measurement_sequence:jax.Array) -> jax.Array:
		""" Corrects predictions on full trajecotry (offline processing)"""
		...

@runtime_checkable # remove for C++
class StateEstimator(Protocol):
	"""
	Forces typehinting and abstraction for fitler class (e.g. EKF, Particle Filter) 
	NOT STATELESS (should keep internal state for covariance calculation)
	"""
	def set_predictor(self, predictor:Predictor) -> None:
		""" sets the predictor for the filter"""
		...
	
	def set_corrector(self, corrector:Corrector) -> None:
		""" sets the corrector or the filter"""
		...

	def update(self, measurement:jax.Array, control:jax.Array, dt:float) -> jax.Array:
		""" predicts and corrects for next time step"""
		...

	# RESERVED FOR FUTURE USE
	def copy(self) -> 'StateEstimator':
		""" returns a memory safe copy of the state estimator"""
		...

