import numpy as np
from reservoirpy.datasets import santafe_laser, narma


# def narmax(order: int, num_timesteps: int=2000, discard: int=20):
#     """
#     Creates NARMA-X sequence for t timesteps, where X is the order of the system.
#     """
#     num_timesteps += discard
#     # input
#     u = np.random.uniform(0, 0.5, (num_timesteps+order, 1)).astype(np.float64)
#     y = narma(n_timesteps=num_timesteps, order=order, u=u)
#     # discard transient effects from first 20 steps
#     u = u[discard+order:]
#     y = y[discard:]
#     return u.T, y.T


def narmax(order: int, num_timesteps: int = 2000, discard: int = 20):
    """
    NARMA-X generator compatible with reservoirpy versions that
    return (input, target) tuples and do NOT support return_input=False.
    """
    import numpy as _np
    from reservoirpy.datasets import narma as _narma

    num_timesteps += discard

    # Create input
    u = _np.random.uniform(0, 0.5, (num_timesteps + order, 1)).astype(_np.float64)

    # Call reservoirpy NARMA
    y = _narma(n_timesteps=num_timesteps, order=order, u=u)

    # --- Handle tuple output (your version) ---
    if isinstance(y, tuple):
        # reservoirpy returns (input, target)
        y = y[1]

    # discard transient effects
    u = u[discard + order:]
    y = y[discard:]

    return u.T, y.T


def santa_fe(num_timesteps: int=2000, discard: int=20):
    """
    Creates NARMA-X sequence for t timesteps, where X is the order of the system.
    """
    num_timesteps += discard
    # input
    u = np.random.uniform(0, 0.5, num_timesteps).astype(np.float64)
    # NARMA sequence
    y = santafe_laser()[:num_timesteps].T

    # discard transient effects from first 20 steps
    u = u[np.newaxis, discard:]
    y = y[:, discard:]

    return u, y


    




