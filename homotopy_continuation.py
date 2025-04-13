import numpy as np
from scipy.optimize import fsolve

def homotopy_continuation(f, x0, t_values):
    """
    Resolve uma equação não linear usando continuação homotópica.

    Args:
        f: Função alvo cuja raiz queremos encontrar.
        x0: Estimativa inicial da solução.
        t_values: Valores de t para a continuação homotópica (de 0 a 1).

    Returns:
        Soluções aproximadas para cada valor de t.
    """
    def H(x, t):
        return (1 - t) * f0(x) + t * f(x)

    # Função simples cuja solução é conhecida
    def f0(x):
        return x - x0

    solutions = []
    x = x0

    for t in t_values:
        x = fsolve(lambda x: H(x, t), x)
        solutions.append(x)
        print(f"t = {t:.2f}, solução = {x}")

    return solutions

# Exemplo de uso
def target_function(x):
    return x**3 - 2*x + 1

x0 = np.array([0.0])  # Estimativa inicial
t_values = np.linspace(0, 1, 100)  # Valores de t de 0 a 1

solutions = homotopy_continuation(target_function, x0, t_values)

print("Soluções finais aproximadas:", solutions[-1])