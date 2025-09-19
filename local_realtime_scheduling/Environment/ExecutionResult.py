# import pickle
# import os
# from typing import List


class Operation_result:
    def __init__(self, job_id, operation_id, **kwargs):
        # self.type = op_type
        self.job_id = job_id
        self.operation_id = operation_id
        self.actual_start_transporting_time = None
        self.actual_finish_transporting_time = None
        self.assigned_transbot = None
        self.actual_start_processing_time = None
        self.actual_finish_processing_time = None
        self.assigned_machine = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Operation(Job_ID={self.job_id}, Operation_ID={self.operation_id}, {vars(self)})"


class Local_Job_result:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = {}
        # self.actual_start_time = None
        # self.actual_finish_time = None

    def add_operation_result(self, operation):
        self.operations[operation.operation_id] = operation

    def __repr__(self):
        return f"Job(Job_ID={self.job_id}, Operations={self.operations})"


class LocalResult:
    def __init__(self):
        self.jobs = {}
        self.actual_local_makespan = None
        self.time_window_start = None
        # self.time_window_end = None

    def add_job_result(self, job):
        self.jobs[job.job_id] = job

    def __repr__(self):
        return f"LocalSchedule(Jobs={self.jobs})"