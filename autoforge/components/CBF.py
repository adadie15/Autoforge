import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from ..core.protocols import BarrierGeometry

@register_pytree_node_class
class DECBF:
	"""Generates a discrete exponential control barrier function for controller optimization. 
	This class will accept a position, radius, shape, and alpha value and generate a linearized representation for CVXPY.
	Fields
		Private _alpha  		- float, gradient/softening for barrier function
		Private _geometry		- BarrierGeometry, object defining the barrier's geometry
		Private _heartbeat 		- float, controls simulation clock (ideally a system constant for synchronization)
		Private _velocity		- jnp.ndarray, velocity vector (x,y)
		Private _dt_controller	- float, time between controller updates
		Private _dt_physics		- float, time between simulated continuous physics update
		Private _name			- string, name of cbf for manipulation 

	Methods
		Public linearize() 		- Linearizes the barrier function
		Public get_alpha() 		- returns the softening value
		Public set_alpha() 		- sets the softening value
		Public get_heartbeat() 	- gets the control simulation clock timing
		Public set_heartbeat() 	- sets the control simulation clock timing
		Public get_shape()		- gets barrier shape # RESERVED FOR FUTURE USE (assumed circular for the moment)
		public set_shape()		- sets barrier shape # RESERVED FOR FUTURE USE (assumed circular for the moment)

	Future Efforts:
		Input Validation
		Generalizing Geometry
		Separate and Modularize Obstacle Dynamic Models
		Add Stochastic Elements
"""
	# Field definitions (example)
	#_alpha: float
	#_radius: float
	#_center: jnp.ndarray
	#_heartbeat: float

	def __init__(self, name:str = None, geometry:BarrierGeometry = None, 
			  heartbeat: float = 0.0, physics_clock: float = 0.0, 
			  alpha: float = 0.0, velocity:jax.Array = None):
		
		# Setting fields
		self._name = name
		self._alpha = alpha
		self._physics_clock = physics_clock
		self._heartbeat = heartbeat

		# Setting default values with checking
		if (geometry is not None):
			self._geometry = geometry.copy()
			dim = self._geometry.get_center().shape[0]
		else:
			self._geometry = None 
			dim = 2

		if (velocity is not None):
			self._velocity = jnp.array(velocity).flatten()
		else:
			self._velocity = jnp.zeros((dim,))

		if (name is not None):
			self._name = name
		else:
			self._name = None
	
		# Setting clocks
		if (self._physics_clock > 0.0):
				self._dt_physics = 1/self._physics_clock
		else:
			self._dt_physics = 0.0

		if (self._heartbeat > 0.0):
			self._dt_controller = 1/self._heartbeat
		else:
			self._dt_controller = 0.0


	# PYTREE FUNCTIONS FOR JAX NOT C++ PROTOTYPE
	def tree_flatten(self):
		"""tells jax which fields are dynamic arrays vs. static metadata"""

		children = (
				self._velocity,
				self._geometry,                          
				jnp.array(self._alpha),
				jnp.array(self._dt_controller),
				jnp.array(self._heartbeat),
				jnp.array(self._physics_clock),
				)
		
		aux_data = ()                     
		return (children, aux_data)
		
	@classmethod
	def tree_unflatten(cls, aux_data, children):
		# repacking variables after vmapping
		obj = cls.__new__(cls)
		obj._name = aux_data[0]
		(obj._velocity, obj._geometry, 
	 	alpha, dt_controller, heartbeat, physics_clock) = children
		obj._alpha          = float(alpha)
		obj._dt_controller  = float(dt_controller)
		obj._heartbeat      = float(heartbeat)
		obj._physics_clock  = float(physics_clock)
		obj._dt_physics     = float(dt_controller)  #UNSYNC LATER WHEN RUNNING WORLD/CONTROLLER CLOCKS
		return obj


	def get_name(self) -> str: return f"{self._name}"
	def set_name(self, name:str) -> None: self._name = f"{name}"

	def get_geometry(self) -> BarrierGeometry: 
		if (self._geometry is None):
			raise ValueError("Geometry uninitialized.")
		return self._geometry.copy()
	
	def set_geometry(self, geometry:'BarrierGeometry') -> None: self._geometry = geometry.copy()

	def get_heartbeat(self) -> float: 
		return float(self._heartbeat) # generates memory isolated copy
	
	def set_heartbeat(self, heartbeat:float) -> None: 
		self._heartbeat = heartbeat
		self._dt_controller = 1 / self._heartbeat

	def get_alpha(self) -> float: 
		return float(self._alpha) # generates memory isolated copy
	
	def set_alpha(self, alpha:float) ->	None: 
		
		if (alpha > 0 and alpha < 1):
			self._alpha = alpha
		else:
			raise ValueError(f"alpha is {alpha}. alpha must be greater than 0 and less than 1")

	def get_velocity(self) -> jax.Array: 
		return self._velocity #IMMUTABLE, MAKE MEMORY SAFE IN C++

	def set_velocity(self, velocity:jax.Array) -> None: 
		self._velocity = jnp.array(velocity).flatten() #IMMUTABLE, MAKE MEMORY SAFE IN C++ 

	def get_physics_clock(self) -> float: return float(self._physics_clock) # generates memory isolated copy
	def set_physics_clock(self, physics_clock:float) -> None: 
		self._physics_clock = physics_clock
		self._dt_physics = 1/ self._physics_clock


	def update_state(self) -> None:
		"""
		Evolves the barrier function's state forward in time for dynamic trajectories
		Params:
			None #IN THE FUTURE INCLUDE A WORLD_CLOCK OR PHYSICS TO CALCULATE ROBUSTLY WITH CLOCKSPEED JITTER
		Returns:
			None
		"""

		if (self._geometry is None):
			raise MemoryError("No geometry set. Set geometry before updating.")
		
		# NOTE: THIS NEEDS TO MANAGE A DYNAMICS OBJECT WHICH DESCRIBES THE EXACT MODEL
		# When that update occurs, physics clock updates will no longer be linear
		# this is constant velocity for simplicity, but more complex dynamics require
		# iterative update for accurate traversal between clock cycles
		self._dt_physics = self._dt_controller #CHANGE TO ALLOW DISSIMILAR CLOCKS IN FUTURE UPDATE
		temp_center = self._geometry.get_center() + self._velocity*self._dt_physics
		self._geometry.set_center(temp_center)

	def linearize(self, p_agent:jax.Array) -> tuple[jax.Array, jax.Array]:
		"""
		Returns linearized barrier function coeffieicnts as jax arrays.

		Params:
			p_agent - jnp.ndarray, cartesian x,y position of autonomous object
		Returns:
			G, b - tuple[jax.Array, jax.Array], coefficients for Gp_next + b >= 0
		"""
		def compute_h(p):
			return self._geometry.get_h(p)
		
		# Converting to numpy for CVXPY with memory isolation
		G = jax.grad(compute_h)(p_agent)

		b = self._alpha * self._dt_controller * compute_h(p_agent)
		b -= jnp.dot(G,p_agent)
		b -= jnp.dot(G,self._velocity) * self._dt_controller


		# Internal expression for data isolation and logging #RESERVED FOR FUTURE USE
		internal_expression = (G, b)

		# External Return
		return G, b #IMMUTABLE, MAKE MEMORY SAFE IN C++