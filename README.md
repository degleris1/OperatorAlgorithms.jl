# OperatorAlgorithms.jl

Developing and testing inverse-free algorithms for optimization.


## Model

`OperatorAlgorithms.jl` supports `NLPModels` models, which have the form,
```
min_u		f(u)
such that	u0 <= u <= u1
		c0 <= c(u) <= c1
```
These problems are then converted into a standard form.
Specifically, we introduce a slack variable `s` and define `x = [u; s]`, then solve,
```
min_x		g(x)
such that	h(x) = 0
		x0 <= x <= x1
```
where `g([u; s]) = f(u)` and `h([u; s]) = c(u) - s`.
The bounds are similarly `x0 = [u0; c0]` and `x1 = [u1; s1]`.
Finally, we rewrite the above problem as
```
min_x		g(x) + I(B, x)
such that	h(x) = 0
```
where `I_B(x)` is the indicator function on the box `[x0, x1]`.
The equality constraint is assigned a dual variable `y`.
