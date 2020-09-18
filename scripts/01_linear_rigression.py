import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 1000)[:, None]
y = 0.5 + 0.1 * x + np.random.normal(scale=0.01, size=(1000, 1))

with plt.xkcd():
    plt.scatter(x, y)
    plt.show()


params = dict(
    w=np.random.uniform(-1, -1, size=(1, 1)),
    b=0.0,
)


def linear(params, x):
    return jnp.dot(x, params["w"]) + params["b"]


def loss_fn(params, x, y):
    y_pred = linear(params, x)
    return jnp.mean((y - y_pred) ** 2)


def sgd(param, grad):
    return param - 0.1 * grad


@jax.jit
def update(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = jax.tree_multimap(sgd, params, grads)
    return loss, new_params


for step in range(1000):
    loss, params = update(params, x, y)

    if step % 100 == 0:
        print(f"{step} - Loss: {loss}")

y_pred = linear(params, x)

with plt.xkcd():
    plt.title(f"y = {float(params['w']):.3f} . x + {float(params['b']):.3f}")
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()
