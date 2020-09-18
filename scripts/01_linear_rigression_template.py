import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 1000)[:, None]
y = 0.5 + 0.1 * x + np.random.normal(scale=0.01, size=(1000, 1))

with plt.xkcd():
    plt.scatter(x, y)
    plt.show()


params = {}
y_pred = ...


with plt.xkcd():
    plt.title(f"y = {float(params['w']):.3f} . x + {float(params['b']):.3f}")
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()
