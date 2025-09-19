from typing import List, Dict
from System.Job import Job
from System.Product import Product


class Order:
    def __init__(self, order_info: Dict[str, int], due_date: float, n_machines: int):
        """
        Initialize an Order with product information and due date.
        :param product_info: A dictionary where keys are product types and values are quantities
        :param due_date: The due date for the order
        """
        self.due_date = due_date
        self.order_info = order_info
        self.products = {product_type: Product(product_type=product_type, n_machines=n_machines) for product_type in self.order_info}
        self.jobs = self.to_jobs()

    def to_jobs(self) -> List[Job]:
        """
        Convert the order into a list of jobs based on product requirements
        :return: A list of jobs
        """
        jobs = []
        job_id = 0
        for product_type, quantity in self.order_info.items():
            product = self.products[product_type]
            for _ in range(quantity):
                job = Job(job_id=job_id, operations_matrix=product.operations_matrix, due_date=self.due_date)
                jobs.append(job)
                job_id += 1
        return jobs

    def get_jobs_summary(self):
        summary = {}
        for job in self.jobs:
            if job.job_id not in summary:
                summary[job.job_id] = {
                    "operations": job.operations_matrix,
                    "due_date": job.due_date
                }
        return summary


# Example usage
if __name__ == "__main__":
    order_info = {
        "Product_A": 1,
        "Product_B": 5,
        "Product_C": 10,
    }
    due_date = 5000
    order = Order(order_info, due_date, 10)
    print(order.get_jobs_summary())
