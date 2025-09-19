import numpy as np
from System.FactoryGraph import FactoryGraph
from System.Machine import Machine
from System.AGV import AGV


class FactoryInstance:
    def __init__(self,
                 seed,
                 n_machines,
                 n_transbots,
                 ):

        np.random.seed(seed)

        self.factory_graph = FactoryGraph(n_machines)

        self.machines = []
        for k in range(n_machines):
            machine_k = Machine(
                machine_id=k,
                failure_threshold=0.3,
                degradation_model={
                    "type": "weibull",
                    "parameters": {
                        "shape": np.random.uniform(low=0.8, high=3.0),
                        "scale": np.random.uniform(low=300, high=600),
                    }
                },
                # location=self.factory_graph.pickup_dropoff_points[f"machine_{k}"]
                location=f"machine_{k}"
            )
            self.machines.append(machine_k)

        self.agv = []
        for k in range(n_transbots):
            agv_k = AGV(agv_id=k)
            self.agv.append(agv_k)


# Example Usage:
if __name__ == "__main__":

    n_machines = 50
    n_transbots = 5

    # Initialize factory instance
    factory_instance = FactoryInstance(
        seed=42,
        n_machines=n_machines,
        n_transbots=n_transbots,
    )

    print(factory_instance.factory_graph.unload_transport_time_matrix)




