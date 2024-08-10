# Predictive Models

## Ethan Caballero's Advice

- Use msle (not mse)
- For 6 parameters or less, just use code on [BNSL GitHub](https://github.com/ethancaballero/broken_neural_scaling_laws) 
- For 30 parameters or less, use random search parallelized on a gpu with jax/PyTorch and then feed result of that into scipy.optimize.curve_fit