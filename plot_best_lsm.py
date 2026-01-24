import sqlite3, jsonpickle
import numpy as np
import matplotlib.pyplot as plt

from measure.tasks import narma
from grow.reservoir import SpikingReservoir   # adjust import if needed
from brian2 import *

DB = "fitness_local_3.db"
RUN_ID = 3
ORDER = 10
WASHOUT = 20


def main():
    conn = sqlite3.connect(DB)
    row = conn.execute(
        "SELECT model, reservoir FROM models WHERE run_id=? AND epoch=-1",
        (RUN_ID,)
    ).fetchone()
    conn.close()

    if row is None:
        raise RuntimeError("No (epoch=-1) row in models table for this run_id.")

    best_dgca = jsonpickle.decode(row[0])       # not used here, but loaded
    best_res = jsonpickle.decode(row[1])        # Reservoir (or SpikingReservoir if you stored that)

    # Wrap reservoir -> LSM
    lsm = SpikingReservoir.from_reservoir(best_res)
    lsm.washout = WASHOUT

    # IMPORTANT: match what fitness does
    lsm = lsm.bipolar()

    # Generate task data
    x, y = narma(2000, order=ORDER)   # shapes: (in_units, T), (out_units, T)
    x = np.asarray(x)
    y = np.asarray(y)

    # Force (units, time)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.shape[0] > x.shape[1]:   # e.g. (T, 1) -> (1, T)
        x = x.T
    if y.ndim == 1:
        y = y.reshape(1, -1)
    elif y.shape[0] > y.shape[1]:
        y = y.T

    # Make lengths match (crop to shortest)
    T = min(x.shape[1], y.shape[1])
    x = x[:, :T]
    y = y[:, :T]
    print("x shape:", x.shape, "y shape:", y.shape)

    # Train + predict (your train returns preds post-washout)
    preds = lsm.train(x, y)
    if preds is None:
        print("preds=None (washout >= T or training failed)")
        return

    w = lsm.washout
    yt = y[0, w:]
    yp = preds[0, :]
    t = np.arange(len(yt))

    plt.figure()
    plt.plot(t, yt, label="target")
    plt.plot(t, yp, label="pred")
    plt.legend()
    plt.title(f"Best evolved reservoir (run_id={RUN_ID})")

    sm = lsm._spikemon

    if len(sm.t) > 0:
        print("Mean firing rate:",
            (len(sm.t) / (lsm.size() * (sm.t[-1] - sm.t[0])) / Hz))
    else:
        print("Mean firing rate: 0.0")
    nmse = np.mean((yt - yp) ** 2) / np.var(yt)
    print(f"NMSE: {nmse:.4f}")

    plt.figure()
    # plt.scatter(sm.t / sm.t.unit, sm.i, s=1)
    plt.plot(sm.t, sm.i, '.k')
    plt.xlabel("t")
    plt.ylabel("neuron")
    plt.title("Spike raster")

    plt.show()
    
if __name__ == "__main__":
    main()