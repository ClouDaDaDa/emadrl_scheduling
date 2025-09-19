import numpy as np
from System.DegradationModel import DegradationModel


class Machine:
    def __init__(self,
                 machine_id: int,
                 degradation_model: dict,
                 location: str,
                 failure_threshold=0.3,):
        """
        :param machine_id: Unique identifier for the machine
        :param degradation_model: The degradation model to be used
        :param location:
        :param failure_threshold: Threshold below which the machine is considered failed

        Maintenance effectiveness coefficients:
        - LM (0): φ = 0.2, reduces virtual time by 20%
        - MM (1): φ = 0.5, reduces virtual time by 50%
        - OH (2): φ = 1.0, complete reset (100% effective)
        - CM (3): φ = 1.0, complete reset (100% effective)
        """

        # Static features of the machine
        self.machine_id = machine_id
        self.location = location
        self.noise_std = 0.0001
        self.failure_threshold = failure_threshold
        # check if degradation_type is valid
        # if ...:
        #     raise ValueError(f"Unknown degradation model type: {degradation_model["type"]}")
        self.degradation_model = DegradationModel(degradation_model["type"], degradation_model["parameters"])
        # maintenance_methods: 0 for "LM", 1 for "MM", 2 for "OH", and 3 for "CM"
        # repair rate is 0.05
        self.maintenance_durations = [4.0, 10.0, 20.0, 25.0]
        self.maintenance_effects = [0.2, 0.5, 1.0, 1.0]

        # Dynamic global features of the machine
        self._init_machine_dynamic_global_features()

        # Dynamic local features of the machine
        self._init_machine_dynamic_local_features()

    def _init_machine_dynamic_global_features(self):
        self.reliability = 1.0  # Initial reliability
        self.next_reliability = self.reliability
        # 0: idle, 1: processing, 2: under maintenance, 3: failed
        self.machine_status = 0
        self.failed = False
        # True machine life, including all waiting time, processing time and maintenance time
        self.cumulative_total_time = 0.0
        # True machine working time, including all processing time
        self.cumulative_work_time = 0.0
        #
        self.dummy_total_time = 0.0
        # Virtual machine working time, including all accumulated processing time since the last maintenance
        self.dummy_work_time = 0.0
        self.cumulative_tasks = 0
        self.scheduled_results = []
        self.reliability_history = {self.cumulative_total_time: self.reliability}
        self.current_processing_task = None  # to process
        self.current_maintenance_method = None
        self.start_time_of_the_task = None
        self.actual_processing_duration = None  # this value is hidden to the agent
        self.estimated_remaining_time_to_finish = 0.0

    def _init_machine_dynamic_local_features(self):
        self.processing_tasks_queue_for_current_time_window = []

    def update_reliability_history(self, time_point: float, reliability: float):
        self.reliability_history[time_point] = reliability

    def is_failed(self) -> bool:
        """
        Check if the machine has failed based on reliability and random shocks
        :return: True if the machine has failed, False otherwise
        """
        if self.reliability < self.failure_threshold:
            self.machine_status = 3
            self.failed = True
            self.estimated_remaining_time_to_finish = 1e8
            return True
        # random_shock_failure
        # todo: the probability should relevant to reliability instead of dummy_work_time
        if np.random.random() < self.degradation_model.failure_rate(current_life=self.dummy_work_time):
            self.machine_status = 3
            self.failed = True
            self.reliability = 0.0
            self.estimated_remaining_time_to_finish = 1e8
            return True
        else:
            return False

    def update_waiting_process(self, waiting_time=1.0):
        self.cumulative_total_time += waiting_time
        self.dummy_total_time += waiting_time

    def start_processing(self,
                         start_time: float,
                         actual_processing_duration: float,
                         estimated_processing_duration: float):
        if start_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {start_time} != {self.cumulative_total_time}")
        self.machine_status = 1  # Processing
        self.start_time_of_the_task = start_time
        self.actual_processing_duration = actual_processing_duration
        self.estimated_remaining_time_to_finish = estimated_processing_duration
        # (task type, task id, start time)
        self.scheduled_results.append(("Processing", self.current_processing_task, start_time))
        # As long as this operation starts, the progress will increase by one.
        self.cumulative_tasks += 1

    def update_degradation_process(self, degrading_time=1.0):
        # reliability (without noise) is 1.0 - degradation
        self.dummy_work_time += degrading_time
        reliability = 1.0 - self.degradation_model.degradation_function(current_life=self.dummy_work_time)
        noise = np.random.normal(0, self.noise_std)
        # noise = 0.0
        self.reliability = max(0.0, min(1.0, reliability + noise))
        self.cumulative_total_time += degrading_time
        self.cumulative_work_time += degrading_time
        self.dummy_total_time += degrading_time
        self.estimated_remaining_time_to_finish -= degrading_time
        self.update_reliability_history(self.cumulative_total_time, self.reliability)

    def update_processing_without_degradation(self, processing_time=1.0):
        self.cumulative_total_time += processing_time
        self.cumulative_work_time += processing_time
        self.dummy_total_time += processing_time
        self.dummy_work_time += processing_time
        self.estimated_remaining_time_to_finish -= processing_time


    def finish_processing(self, finish_time: float):
        if finish_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {finish_time} != {self.cumulative_total_time}")
        self.machine_status = 0  # Idle
        self.start_time_of_the_task = None
        self.actual_processing_duration = None
        self.estimated_remaining_time_to_finish = 0.0
        # (task type, task id, finish time)
        self.scheduled_results.append((self.scheduled_results[-1][:2] + (finish_time,)))
        self.current_processing_task = None
        self.is_failed()

    def start_maintenance(self, start_time: float):
        if start_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {start_time} != {self.cumulative_total_time}")
        if self.current_maintenance_method < 0 or self.current_maintenance_method > 3:
            raise ValueError(f"Unknown maintenance method: {self.current_maintenance_method}")
        self.machine_status = 2  # Maintenance
        self.failed = False
        self.start_time_of_the_task = start_time
        self.actual_processing_duration = self.maintenance_durations[self.current_maintenance_method]
        self.estimated_remaining_time_to_finish = self.maintenance_durations[self.current_maintenance_method]
        self.scheduled_results.append(("Maintenance", self.current_maintenance_method, start_time))
        # self.update_reliability_history(start_time, self.reliability)

    def update_maintenance_process(self, repairing_time=1.0):
        self.reliability = min(1.0, self.reliability + 0.05 * repairing_time)
        self.cumulative_total_time += repairing_time
        self.estimated_remaining_time_to_finish -= repairing_time
        self.update_reliability_history(self.cumulative_total_time, self.reliability)

    def finish_maintenance(self, finish_time: float):
        if finish_time != self.cumulative_total_time:
            raise Exception(f"Error in time synchronization with the environment: {finish_time} != {self.cumulative_total_time}")
        self.machine_status = 0  # Idle
        self.dummy_total_time = 0.0

        self.dummy_work_time = 0.0

        self.start_time_of_the_task = None
        self.actual_processing_duration = None
        self.estimated_remaining_time_to_finish = 0.0
        self.scheduled_results.append((self.scheduled_results[-1][:2] + (finish_time,)))
        self.current_maintenance_method = None

    def reset_machine(self):
        # Dynamic global features of the machine
        self._init_machine_dynamic_global_features()

        self.reset_machine_for_current_time_window()

    def reset_machine_for_current_time_window(self):
        # Dynamic local features of the machine
        self._init_machine_dynamic_local_features()


# Example usage
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    num_machines = 5
    machines = []
    reliabilities = []
    for k in range(num_machines):
        machine_k = Machine(
            machine_id=k,
            location=f"machine_{k}",
            failure_threshold=0.3,
            degradation_model={
                "type": "weibull",
                "parameters": {
                    "shape": np.random.uniform(low=0.8, high=3.0),
                    "scale": np.random.uniform(low=500, high=3000),
                }
            }
        )
        machines.append(machine_k)
        reliabilities.append([])

    # Simulate some steps and collect reliability values
    num_steps = 8000
    for step in range(num_steps):
        delta_t = 1.0
        for machine in machines:
            machine.update_degradation_process()
            # machine.degradation_process(job_id=0,
            #                             current_reliability=machine.reliability,
            #                             start_time=machine.cumulative_total_time,
            #                             delta_time=delta_t)
            if machine.machine_status == 2:
                machine.update_maintenance_process()
            if machine.is_failed():
                print(f"Step {step}: Machine {machine.machine_id} has failed! Performing corrective maintenance.")
                machine.start_maintenance(maintenance_method=3, start_time=step)
            else:
                print(f"Step {step}: Reliability of Machine {machine.machine_id} is {machine.reliability}")
            reliabilities[machine.machine_id].append(machine.reliability)

    # Plot the reliability curve
    plt.figure(figsize=(10, 6))
    for k in range(num_machines):
        plt.plot(reliabilities[k], label=f"Reliability_{k}")
    plt.xlabel("Step")
    plt.ylabel("Reliability")
    plt.title("Machine Reliability Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


