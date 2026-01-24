import sqlite3, jsonpickle
from evolve.mga import EvolvableDGCA
from grow.reservoir import Reservoir
from helloDGCA import draw_reservoir


conn = sqlite3.connect("fitness_local_4.db")

# fetch the final (best) model — epoch = -1 is used for "final best"
row = conn.execute(
    "SELECT model, reservoir FROM models WHERE run_id=? AND epoch=-1",
    (2,)
).fetchone()

conn.close()

# decode back into Python objects
best_dgca = jsonpickle.decode(row[0])       # EvolvableDGCA instance
best_reservoir = jsonpickle.decode(row[1])  # Reservoir instance

print("Best reservoir size:", best_reservoir.size())
print("Edges:", int(best_reservoir.A.sum()))




draw_reservoir(best_reservoir)


# from grow.reservoir import get_seed
# seed = get_seed(0, 0, best_reservoir.n_states)
# res = seed
# for step in range(200):
#     res = best_dgca.step(res)
#     print(f"Step {step+1}: {res.size()} nodes")
#     draw_reservoir(res, title=f"step {step}")
#     if res.size() >= best_reservoir.size():
#         break