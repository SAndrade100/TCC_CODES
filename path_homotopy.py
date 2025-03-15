import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f0(x):
    """Função inicial: f(x) = x^2"""
    return x**2

def f1(x):
    """Função final: f(x) = sin(x) + 1"""
    return np.sin(x) + 1

def homotopia(x, t):
    """
    Função de homotopia que deforma f0 em f1
    
    Parâmetros:
    x: ponto do domínio
    t: parâmetro da homotopia (0 <= t <= 1)
       t=0 corresponde a f0
       t=1 corresponde a f1
    """
    return (1-t) * f0(x) + t * f1(x)

# Configurações para visualização
x = np.linspace(-2, 2, 1000)
t_valores = np.linspace(0, 1, 50)  # 50 frames para a animação

# Criar figura e eixos
fig, ax = plt.subplots(figsize=(10, 6))
linha, = ax.plot([], [], 'b-', lw=2)
titulo = ax.set_title('Homotopia: t = 0.00')
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 4)
ax.set_xlabel('x')
ax.set_ylabel('H(x, t)')
ax.grid(True)

# Adicionar as funções iniciais e finais como referência
ax.plot(x, f0(x), 'r--', alpha=0.5, label='f₀(x) = x²')
ax.plot(x, f1(x), 'g--', alpha=0.5, label='f₁(x) = sin(x) + 1')
ax.legend()

# Função de inicialização para a animação
def init():
    linha.set_data([], [])
    return linha,

# Função de atualização para a animação
def update(frame):
    t = t_valores[frame]
    y = homotopia(x, t)
    linha.set_data(x, y)
    titulo.set_text(f'Homotopia: t = {t:.2f}')
    return linha, titulo

# Criar a animação
ani = FuncAnimation(fig, update, frames=len(t_valores), init_func=init, 
                   blit=True, interval=100)

# Para salvar a animação (opcional)
# ani.save('homotopia.gif', writer='pillow', fps=10)

plt.show()

# Função para visualizar algumas etapas específicas da homotopia
def visualizar_etapas(num_etapas=5):
    """Visualiza algumas etapas específicas da homotopia."""
    fig, axs = plt.subplots(1, num_etapas, figsize=(15, 4))
    
    for i in range(num_etapas):
        t = i / (num_etapas - 1)
        y = homotopia(x, t)
        axs[i].plot(x, y, 'b-')
        axs[i].plot(x, f0(x), 'r--', alpha=0.3)
        axs[i].plot(x, f1(x), 'g--', alpha=0.3)
        axs[i].set_title(f't = {t:.2f}')
        axs[i].set_xlim(-2, 2)
        axs[i].set_ylim(-0.5, 4)
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso
visualizar_etapas(5)

# Versão alternativa: Homotopia não-linear (para resultados mais interessantes)
def homotopia_nao_linear(x, t):
    """
    Versão alternativa usando uma homotopia não-linear
    que pode produzir deformações mais interessantes
    """
    # Usando uma função sigmoide para controlar a transição
    sigmóide = 1 / (1 + np.exp(-10 * (t - 0.5)))
    return (1 - sigmóide) * f0(x) + sigmóide * f1(x)

# Para testar a versão não-linear, substitua a função homotopia pela 
# homotopia_nao_linear nas funções de animação acima