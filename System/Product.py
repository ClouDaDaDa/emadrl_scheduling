from typing import Optional
import numpy as np

PRODUCTS_DICT = {
    "Product_A": 10,
    "Product_B": 10,
    "Product_C": 10,
    "Product_D": 10,
    "Product_E": 10,
    "Product_F": 15,
    "Product_G": 15,
    "Product_H": 15,
    "Product_I": 15,
    "Product_J": 15,
    "Product_K": 20,
    "Product_L": 20,
    "Product_M": 20,
    "Product_N": 20,
    "Product_O": 20,
    "Product_P": 25,
    "Product_Q": 25,
    "Product_R": 25,
    "Product_S": 25,
    "Product_T": 25,
    "Product_U": 30,
    "Product_V": 30,
    "Product_W": 30,
    "Product_X": 30,
    "Product_Y": 30,
    "Product_Z": 35
}


class Product:
    def __init__(self, product_type: str, n_machines: int = 10, unprocessable_prob: float = 0.2,
                 seed: Optional[int] = None):
        """
        :param product_type: The type of the product
        :param n_machines: Number of machines (default is 10)
        :param unprocessable_prob: Probability that an operation cannot be processed by a machine (default is 0.2)
        :param seed: Random seed for reproducibility (default is None)
        """
        if product_type not in PRODUCTS_DICT:
            raise ValueError(f"Unknown product type {product_type}!")

        self.product_type = product_type
        self.num_ops = PRODUCTS_DICT[self.product_type]
        self.n_machines = n_machines
        self.unprocessable_prob = unprocessable_prob

        if seed is not None:
            np.random.seed(seed)

        self.operations_matrix = self.get_operations(num_ops=self.num_ops)

    def get_operations(self, num_ops: int) -> np.ndarray:
        operations_matrix = np.random.randint(low=1, high=100, size=[num_ops, self.n_machines])

        # Randomly assign -1 to indicate operation cannot be processed by the machine
        mask = np.random.choice([0, 1], size=operations_matrix.shape, p=[self.unprocessable_prob, 1 - self.unprocessable_prob])
        operations_matrix = np.where(mask == 0, -1, operations_matrix)

        return operations_matrix

    def display_operations(self):
        print(f"Operations for product type {self.product_type}:")
        print(self.operations_matrix)


# Example usage
if __name__ == "__main__":
    product_1 = Product("Product_A", seed=42)
    product_1.display_operations()
    product_2 = Product("Product_B", n_machines=12, unprocessable_prob=0.3)
    product_2.display_operations()
    product_3 = Product("Product_S", seed=41)
    product_3.display_operations()
    try:
        product_4 = Product("Product_XXX")
        product_4.display_operations()
    except ValueError as e:
        print(e)
