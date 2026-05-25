from typing import Protocol, Optional, Any, runtime_checkable
import jax


# Dynamics Models List
@runtime_checkable # remove for C++
class DynamicsModel(Protocol):
	"""
	forces typehinting and abstraction of various
	dynamics models stateless
	"""
	
	def evolve(self, state:jax.Array, control:jax.Array, dt:float) -> jax.Array:
		""" Steps dynamic model forward in time by one dt"""
		...

# Observation Models List
@runtime_checkable # remove for C++
class ObservationModel(Protocol):
	"""
	defines an observation model type (e.g. satellite, range/heading, etc)
	"""

	def observe(self, 
			 prediction:jax.Array
			 ) -> jax.Array:
		...

# Probability Distribution List
@runtime_checkable # remove for C++
class LikelihoodModel(Protocol):
	"""
	defines a likelihood methodology given estimated data
	"""

	def log_evaluate(self, 
			  measurement:jax.Array, 
			  predicted_measurement: jax.Array, 
			  covariance: jax.Array,
			  predicted_state: Optional[jax.Array] = None
			  ) -> jax.Array:
		...

# Class List: Estimator Protocols
@runtime_checkable # remove for C++
class Predictor(Protocol):
	"""
	Forces abstraction of predictor classes for state estimation
	"""
	def predict_step(self, 
				  state:jax.Array, 
				  control:jax.Array, 
				  dt:float, 
				  dyanmics:DynamicsModel,
				  key: Optional[jax.Array] = None, 
				  noise_distribution: Optional[Any] = None,
				  standard_dev: Optional[jax.Array] = None,
				  ) -> jax.Array:
		""" Predicts the next time step (online processing, Realtime)"""
		...
	
	def predict_trajectory(self, 
						initial_state:jax.Array,  
						control_sequence:jax.Array, 
						dt:float, 
						dynamics:DynamicsModel,
						keys:Optional[jax.Array] = None,
						noise_distribution: Optional[Any] = None,
						standard_dev: Optional[jax.Array] = None,
						) -> jax.Array:
		"""Predictions on full trajectory (offline processing)"""
		...

@runtime_checkable # remove for C++
class Corrector(Protocol):
	"""
	Forces abstraction of corrector classes for state estimation
	"""
	def correct_step(self,
			weights: Optional[jax.Array], 
			particles: Optional[jax.Array],
			likelihood: Optional[LikelihoodModel],
			measurements: jax.Array,
			observation_model: ObservationModel,
			covariance: jax.Array
			) -> jax.Array:
		""" Corrects prediction of the next time step (online processing, Realtime)"""
		...

	def correct_trajectory(
						weights: jax.Array, 
						particles: Optional[jax.Array],
						likelihood: Optional[LikelihoodModel],
						measurement_sequence: jax.Array,
						observation_model: ObservationModel,
						covariance: jax.Array
						) -> jax.Array:
		""" Corrects predictions on full trajecotry (offline processing)"""
		...

@runtime_checkable # remove for C++
class StateEstimator(Protocol):
	"""
	Forces typehinting and abstraction for fitler class (e.g. EKF, Particle Filter) 
	Stateful (should keep internal state for covariance calculation)
	"""

	def estimate(self) -> jax.Array:
		""" predicts and corrects for next time step"""
		...

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

