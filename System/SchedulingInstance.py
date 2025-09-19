import numpy as np
from configs import dfjspt_params
from System.Job import Job


class SchedulingInstance:
    def __init__(self,
                 seed,
                 n_jobs,
                 n_machines,
                 ):

        if seed is None:
            self.n_jobs_for_this_instance = None
            self.n_operations_for_jobs = None
            self.job_arrival_time = None
            self.job_due_date = None
            self.jobs = []
        else:
            np.random.seed(seed)

            # Whether the number of jobs in this scheduling instance is fixed (n_jobs) or random
            if dfjspt_params.n_jobs_is_fixed:
                self.n_jobs_for_this_instance = n_jobs
            else:
                self.n_jobs_for_this_instance = np.random.randint(10, n_jobs + 1)

            # Whether the number of operations for each job is fixed (n_machines) or random
            if dfjspt_params.n_operations_is_n_machines:
                self.n_operations_for_jobs = n_machines * np.ones((self.n_jobs_for_this_instance,), dtype=int)
            else:
                self.n_operations_for_jobs = np.random.randint(low=dfjspt_params.min_n_operations,
                                                               high=dfjspt_params.max_n_operations + 1,
                                                               size=self.n_jobs_for_this_instance, dtype=int)

            self.job_arrival_time = np.zeros(shape=(n_jobs,), dtype=float)
            if dfjspt_params.consider_job_insert and dfjspt_params.new_arrival_jobs > 0:
                for new_job in range(dfjspt_params.new_arrival_jobs):
                    self.job_arrival_time[n_jobs - new_job - 1] = np.random.randint(
                        low=dfjspt_params.earliest_arrive_time,
                        high=dfjspt_params.latest_arrive_time + 1
                    )

            self.job_due_date = 1e8 * np.ones(shape=(n_jobs,), dtype=float)

            # processing time for each operation
            processing_time_baseline = np.random.randint(low=dfjspt_params.min_prcs_time,
                                                         high=dfjspt_params.max_prcs_time,
                                                         size=(n_jobs, max(self.n_operations_for_jobs)))
            processing_time_matrix = -1 * np.ones((n_jobs, max(self.n_operations_for_jobs), n_machines), dtype=float)
            if dfjspt_params.is_fully_flexible:
                n_compatible_machines_for_operations = n_machines * np.ones(shape=(n_jobs, max(self.n_operations_for_jobs)),
                                                                            dtype=int)
            else:
                n_compatible_machines_for_operations = np.random.randint(dfjspt_params.min_compatible_machines, n_machines + 1,
                                                                         size=(n_jobs, max(self.n_operations_for_jobs)))

            for job_id in range(n_jobs):
                for operation_id in range(self.n_operations_for_jobs[job_id]):
                    if dfjspt_params.time_for_compatible_machines_are_same:
                        processing_time_matrix[job_id, operation_id, :] = processing_time_baseline[job_id, operation_id]
                    else:
                        # the same operation may have different time in different machines
                        processing_time_matrix[job_id, operation_id, :] = np.random.randint(
                            max(dfjspt_params.min_prcs_time,
                                processing_time_baseline[job_id, operation_id] - dfjspt_params.time_viration_range),
                            min(dfjspt_params.max_prcs_time,
                                processing_time_baseline[job_id, operation_id] + dfjspt_params.time_viration_range + 1),
                            size=(n_machines,)
                        )
                    # some machine may cannot process this operation
                    zero_columns = np.random.choice(n_machines,
                                                    n_machines - n_compatible_machines_for_operations[job_id, operation_id],
                                                    replace=False)
                    processing_time_matrix[job_id, operation_id, zero_columns] = -1

            self.jobs = []
            for job_id in range(n_jobs):
                job = Job(job_id=job_id,
                          operations_matrix=processing_time_matrix[job_id][:self.n_operations_for_jobs[job_id]],
                          arrival_time=self.job_arrival_time[job_id],
                          due_date=self.job_due_date[job_id])
                self.jobs.append(job)


# Example Usage:
if __name__ == "__main__":

    n_jobs = dfjspt_params.n_jobs
    n_machines = dfjspt_params.n_machines
    # n_transbots = 5

    # Initialize scheduling instance
    scheduling_instance = SchedulingInstance(
        seed=52+dfjspt_params.current_scheduling_instance_id,
        n_jobs=n_jobs,
        n_machines=n_machines,
    )

    print(scheduling_instance)



