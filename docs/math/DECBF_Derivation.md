## Discrete Exponential CBF Definition (dynamic and static case, circular barriers)
Defining CBF as our control barrier functions around (linearized) 

$$ h(p_{t+1}, t+1) $$

$$ Gp_t \ge (1 - \gamma) * h(p_t) $$

$$ \gamma \in (0,1]$$

Taylor Expansion:

$$h(p_{t+1}, t+1) \approx h(p_t,t) + \frac{\partial h}{\partial p}(p_{t+1}-p_t) + \frac{\partial h}{\partial t} dt $$

By Substitution:

$$ \frac{\partial h}{\partial p} p_{t+1} + [\gamma h(p_t,t) - \frac{\partial h}{\partial p}p_t + \frac{\partial h}{\partial t}dt] \ge 0 $$

Using the gradient defined above:

$$ G = 2[p_t - p_{obs,t}]^T $$

Applyng chain rule to temporal element:

$$ \frac{\partial h}{\partial t} = \frac{\partial h}{\partial p_{obs}} \frac{\partial p_{obs}}{\partial t} = -Gv_{obs}$$

$$ \alpha * dt = \gamma $$

Therefore:

$$ \boxed{ Gp_{t+1} + [\gamma h(p_t, t) - Gp_t - Gv_{obs}dt] \ge 0 } $$

NOTE: Citation needed
