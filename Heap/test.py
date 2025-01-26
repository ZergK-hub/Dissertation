def combine_lambda_functions(f_list, mult):
    """
    Combines multiple lambda functions into a single lambda function with multipliers.

    Args:
        f_list (list): A list of lambda functions.
        mult (list): A list of multipliers corresponding to the lambda functions.

    Returns:
        function: A new lambda function that represents the combined function.
        Raises:
          ValueError: if the number of functions is not equal to number of multipliers
    """
    if len(f_list) != len(mult):
      raise ValueError("Number of functions must equal number of multipliers")

    combined_func = lambda x: sum(mult[i] * f_list[i](x) for i in range(len(f_list)))

    return combined_func


# Example usage:
f1 = lambda x: x**2
f2 = lambda x: x**3
f3 = lambda x: x*2

functions = [f1, f2, f3] # List of lambda functions
multipliers = [2, 6, 1]   # List of multipliers

combined_function = combine_lambda_functions(functions, multipliers)


x_value = 3
result = combined_function(x_value)
print(f"Combined function result for x={x_value}: {result}")  # Output: Combined function result for x=3: 171

