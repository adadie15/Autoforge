### Formulation
Particle Filter Estimator Definintion

given $h$, $R$, $f$, $v$

Intialization:

$$ x^i(0) \sim Px(0)$$

$$w^i(0) = \frac{1}{N}$$

Prediction (simulation):

$$x^i(t+1) = f(x^i(t)) + v^i(t)$$

Condition (weighting):

$$w^i(t+1) \propto w_i(t) \gamma_{t+1}(y(t+1) | x^i(t+1))$$

probability function $(\gamma^i_{t+1})$

$$\gamma^i_{t+1} = \exp{(-\frac{1}{2}(y(t+1) - h(x^i(t+1))^TR^{-1}(y(t+1) - h(x^i(t+1)))}$$

Estimation:

$$ \pi_{t+1|t+1} = P_{(x(t+1) | y_{1:t+1})} \approx \sum_{i = 1} ^N w_i(t+1)\delta x^i(t+1)$$
