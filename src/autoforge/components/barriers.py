import jax
import jax.numpy as jnp

from ..core.protocols import BarrierGeometry

class EllipticalGeometry(BarrierGeometry):
	"""
	defines a circular geometry for barrier function creation.
	
	Fields:
		Private _semi_major_axis 	-float, defines the major radius of the geometric object
		Private _semi_minor_axis 	-float, defines the minor radius of the ellipses #RESERVED FOR FUTURE USE
		Private _angle				-float, radians, defines the offset angle of the major axis from x axis #RESERVED FOR FUTURE USE
		Private _center 			-jnp.ndarray, defines the center of the geometric object
		
	Methods:
		Public get_semi_major_axis()	-get major radius value
		Public set_semi_major_axis()	-set major radius value
		Public get_semi_minor_axis()	-get minor radius value #RESERVED FOR FUTURE USE
		Public set_semi_minor_axis()	-set minor radius value #RESERVED FOR FUTURE USE
		Public get_angle()				-get major axis angle from universal x-axis #RESERVED FOR FUTURE USE
		Public set_angle()				-set major axis angle from universal x-axis #RESERVED FOR FUTURE USE
		Public set_relative_angle()		-set major axis angle from relative heading #RESERVED FOR FUTURE USE
		Public get_center()				-get center value
		Public set_center()				-set radius value
		Public get_h()					-defines a geometric definition to autodifferentiate with jax
	"""
	# For reference for C++ conversion
	#_semi_major_axis:float
	#_center:jnp.ndarray
	
    #NOTE: TO BE EXPANDED TO N-DIMENSIONAL ELLIPSES EVALUATION IN THE FUTURE
	
	def __init__(self, center: jax.Array = None, semi_major_axis: float = 1.0):
		if (center is not None):
			self._center = jnp.array(center).flatten() #IMMUTABLE, MAKE MEMORY SAFE IN C++
		else:
			self._center = jnp.zeros((2,))

		self._semi_major_axis = semi_major_axis

	def get_semi_major_axis(self) -> float: 
		return self._semi_major_axis
	
	def set_semi_major_axis(self, semi_major_axis: float): 
		self._semi_major_axis = semi_major_axis
		
	def get_center(self) -> jax.Array: 
		return self._center #IMMUTABLE, MAKE MEMORY SAFE IN C++
	

	def set_center(self, center: jax.Array): 
		self._center = jnp.array(center).flatten() #IMMUTABLE, MAKE MEMORY SAFE IN C++

	def copy(self) -> 'EllipticalGeometry':
		"""Returns a memory-safe copy of this object/class"""
		return EllipticalGeometry(center=self._center, semi_major_axis=self._semi_major_axis)

	def get_h(self, p_agent: jax.Array) -> jax.Array:
		"""Defines the geometry of the barrier for differentiation
		Params:
			p_agent 	- jax.Array, center of the autonomous object
		Returns:
		 	h 		- jax.Array, mathematical definition of the object
		"""
		# ||p_agent - self._center||^2 - self._semi_major_axis^2 = h
		# h is the minimum radius between centers for objects in this definition
		
		# CHANGE FOR ELLIPSE EQUATION IN THE FUTURE, P - shape matrix should be included 
		d_p = p_agent.flatten() - self._center.flatten()
		h = jnp.sum(d_p**2) - self._semi_major_axis**2

		return h
