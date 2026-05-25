import jax
import jax.numpy as jnp
from ..components.CBF import DECBF

# Batching Linearize
# in_axes=(0, None) (map over self)
_batch_linearize = jax.jit(jax.vmap(DECBF.linearize, in_axes=(0, None)))

# Batching Update
# in_axes default (0,) (map over self)
_batch_update = jax.jit(jax.vmap(DECBF.update_state))

class CBFManager:
	"""
	Generates, manages, and in the future eliminates multiple control barrier functions dynamically across time
	Fields
		Private _cbf_buffer					-list, list of cbfs to 
		Private	_MAX_CBFS					-int, max buffer length for tracked cbfs
		Private _cbf_count					-int, number of cbfs currently tracked
		
	Methods
		Public get_cbf_linearizations()		- returns cbf linearization for cvxpy implementation for trajectories
		Public add_cbf()					- creates a new cbf for simulation with initial position, velocity #RESERVED FOR FUTURE USE - ADD DYNAMICS, SHAPE
		Public get_cbf_list()				- returns a list of the cbfs currently in list
		Private _physics_clock_update()		- updates physics clock on CBF objects
		Private _manage_cbf_states()		- updates the states of the cbfs 
	"""

	def __init__(self, max_cbfs: int, template: DECBF):
		# Defining maximum number of CBFs
		self._MAX_CBFS= max_cbfs

		# python layer name tracking
		self._name_registry: dict[int, str] = {}
		
		# boolean mask for memory slots for jax
		self._active_mask = jnp.zeros(self._MAX_CBFS, dtype=bool)

		self._cbf_count = 0

		# jax layer Pre-allocate stack pytree from template
		# treemap recurses to find all leaves to go from shape to
		# a 2d array of (max num, vectorized attributes)
		self._stacked = jax.tree_util.tree_map(
			lambda leaf: jnp.zeros(
        	(max_cbfs, *jnp.asarray(leaf).shape),
        	dtype=jnp.asarray(leaf).dtype
    		),
    		template
		)

	def get_cbf_names(self) -> list[str]:
		"""Returns a python list copying the list of cbfs
		Params:
			None
		Return:
			cbf_names - list, python list of cbf names
		"""
		
		# VECTORIZE IN FUTURE UPDATE
		cbf_names = [""] * self._cbf_count
		for i in range(self._cbf_count):
			cbf_names[i] = self._cbf_buffer[i].get_name()

		return cbf_names
		
	# RESERVED FOR FUTURE UPDATE. FOR NOW, EVERYTHING RUNS ON THE SAME CLOCK
	#def _physics_clock_update():
	#	"""Updates world clock within cbfs"""
	#	...

	#def _manage_cbf_states():
	#	""" Update cbf states according to controller clock"""
	#	...
	
	def _write_slot(self, slot:int, cbf:DECBF) -> None:
		""" Writes a CBF into a specific memory slot"""
		self._stacked = jax.tree_util.tree_map(
			lambda batch, single: batch.at[slot].set(single),
			self._stacked,
			cbf
		)

	def _read_slot(self, slot:int) -> DECBF:
		""" reads and copies a CBF from memory"""
		return jax.tree_util.tree_map(
			lambda batch: batch[slot],
			self._stacked
		)

	def add_cbf(self, name:str, cbf:DECBF) -> None:
		"""Adds a cbf to the manager buffer
		Params:
			geometry 		- BarrierGeometry, geometric definition of barrier
			heatbeat 		- float, controller clockspeed
			physics_clock 	- float, world clockspeed
			alpha			- float, barrier softening function
			velocity		- jnp.ndarray, velocity array
		Returns:
			None
		"""

		#buffer check
		if self._cbf_count >= self._MAX_CBFS:
			raise MemoryError("Buffer overflow: Max CBFS exceeded")
		
		slot = self._cbf_count

		# python tracking
		self._name_registry[slot] = name

		# jax memory adjustment
		self._write_slot(slot, cbf)
		self._active_mask = self._active_mask.at[slot].set(True)

		self._cbf_count += 1

	def remove_cbf(self, name:str) -> None:
		"""removes a cbf to the manager buffer
		Params:
			cbf 		- DECBF, cbf to remove
		Returns:
			None
		"""

		#buffer check
		if self._cbf_count < 1:
			raise MemoryError("No CBFs in buffer. Please add cbf before removing")
		
		target = next(
			(slot for slot, n in self._name_regristry.items() if n==name),
			None
		)
		if target is None:
			raise ValueError(f"No CBF name '{name}' in buffer")
		
		last = self._cbf_count - 1
		
		if target != last:
			# move last slot to target
			self._write_slot(target, self._read_slot(last))
			self._name_registry[target] = self._name_registry[last]
			# target bool mask is true, holding current data

		self._active_mask = self._active_mask.at[last].set(False)
		del self._name_registry[last]
		self._cbf_count -= 1
		# the mask zeros out old data at compute time

	def get_cbf_linearizations_jax(self, p_agent: jax.Array) -> tuple[jax.Array, jax.Array]:
		"""
		Returns G(MAX_CBFS, dim) and b(MAX_CBFS,)
		inactive slots are zeroed - safe to pass directly to QP solver
		"""
		G, b = _batch_linearize(self._stacked, p_agent)

		#mask: inactive rows become zero in G and zer in b
		# [:, None] broadcasts the (MAX_CBFS,) mask to (MAX_CBFS, dim)
		# if mask is 1, then 1*value if mask is False then value becomes None
		return G * self._active_mask[:, None], b*self._active_mask
	
	def step(self) -> None:
		"""Advances all cbfs one step forward"""
		self._stacked = _batch_update(self._stacked)