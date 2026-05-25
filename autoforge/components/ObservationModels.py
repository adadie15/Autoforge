import jax
import jax.numpy as jnp

from ..core.protocols import ObservationModel

class RangeOnlyTrilateration(ObservationModel):
	"""
	Stateless observation model for two sensors on a plane
	Method:
		Public observe() 	- returns a mathematical evaluation for observations
	"""
	
	# NOTE: THIS MUST BE MUCH MORE DYNAMIC, INCLUDE A PARAMETER FOR SENSORS IN THE FUTURE
	# THIS MODEL IS ACCOUNTING FOR SENSORS AT X = +- 1, Y = 0
	@staticmethod
	@jax.jit
	def observe(state:jax.Array):
		"""
		calculates the positional observation given prediction data
		Params
			state		-[x, y, ...] predicted measurements to estimate observaitons (first two elements are x,y)
		Returns
			jnp.array 	-observational model as a vector
		"""
		x = state[0]
		y = state[1]
		y1 = jnp.sqrt((x - 1)**2 + y**2)
		y2 = jnp.sqrt((x + 1)**2 + y**2)

		return jnp.array([y1, y2])