import numpy as np

class DegradationModel:
    def __init__(self, degradation_type: str, degradation_parameters: dict):
        """
        Initialize the degradation model with specified type and parameters.

        :param degradation_type: Type of degradation model ('weibull', 'lognormal', 'gamma', 'fatigue', etc.)
        :param degradation_parameters: Dictionary of parameters specific to the chosen degradation model.
        """
        self.degradation_type = degradation_type
        self.degradation_parameters = degradation_parameters

    def degradation_function(self, current_life: float) -> float:
        """
        Calculate the degradation level of the machine at a given time (current_life).

        :param current_life: Current life or time point to calculate degradation level.
        :return: Degradation level (unreliability) at the given time point.
        """
        if self.degradation_type == "weibull":
            shape = self.degradation_parameters["shape"]
            scale = self.degradation_parameters["scale"]
            return 1 - np.exp(- (current_life / scale) ** shape)

        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def failure_rate(self, current_life: float) -> float:
        """
        Calculate the instantaneous failure rate (hazard rate) of the machine at a given time.

        :param current_life: Current life or time point to calculate failure rate.
        :return: Failure rate (instantaneous probability of failure at the given time).
        """

        if current_life == 0.0:
            return 0.0
        else:
            if self.degradation_type == "weibull":
                shape = self.degradation_parameters["shape"]
                scale = self.degradation_parameters["scale"]
                return (shape / scale) * (current_life / scale) ** (shape - 1)

            else:
                raise ValueError(f"Unknown degradation type: {self.degradation_type}")


def calculate_current_life(shape, scale, degradation_target=0.7):

    target_degradation = degradation_target

    ln_value = np.log(1 - target_degradation)

    current_life = scale * (abs(ln_value)) ** (1 / shape)
    return current_life


# Example usage:
if __name__ == "__main__":

    shape = np.random.uniform(0.8, 3.0)
    scale = np.random.uniform(500, 1000)
    # shape = 3.0
    # scale = 3000

    print(f"shape = {shape}")
    print(f"scale = {scale}")

    current_life = calculate_current_life(shape, scale)
    print(f"Calculated current life: {current_life:.2f} units")





