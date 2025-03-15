import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import random

class TSPHomotopia:
    def __init__(self, num_cidades=10, seed=42):
        """
        Inicializa o problema do caixeiro viajante com homotopia.
        
        Parâmetros:
        num_cidades: Número de cidades
        seed: Semente para reprodutibilidade
        """
        self.num_cidades = num_cidades
        np.random.seed(seed)
        random.seed(seed)
        
        # Gerar cidades aleatórias em um plano 2D
        self.cidades = np.random.rand(num_cidades, 2)
        
        # Calcular matriz de distâncias (simétrica)
        self.matriz_distancias = squareform(pdist(self.cidades, 'euclidean'))
        
        # Gerar uma rota inicial aleatória
        self.rota_inicial = list(range(num_cidades))
        random.shuffle(self.rota_inicial)
        self.rota_inicial.append(self.rota_inicial[0])  # Retornar à cidade inicial
        
        # Gerar uma rota melhorada via algoritmo guloso (nearest neighbor)
        self.rota_final = self.nearest_neighbor()
        self.rota_final.append(self.rota_final[0])  # Retornar à cidade inicial

    def nearest_neighbor(self):
        """Implementa o algoritmo do vizinho mais próximo para gerar uma rota melhorada."""
        cidades_não_visitadas = set(range(self.num_cidades))
        cidade_atual = random.choice(list(cidades_não_visitadas))
        rota = [cidade_atual]
        cidades_não_visitadas.remove(cidade_atual)
        
        while cidades_não_visitadas:
            # Encontrar a cidade não visitada mais próxima
            próxima_cidade = min(cidades_não_visitadas, 
                               key=lambda cidade: self.matriz_distancias[cidade_atual][cidade])
            rota.append(próxima_cidade)
            cidades_não_visitadas.remove(próxima_cidade)
            cidade_atual = próxima_cidade
            
        return rota
    
    def calcular_comprimento_rota(self, rota):
        """Calcula o comprimento total de uma rota."""
        return sum(self.matriz_distancias[rota[i]][rota[i+1]] 
                  for i in range(len(rota)-1))
        
    def interpolar_rota(self, t):
        """
        Implementa a homotopia entre a rota inicial e a rota final.
        
        A interpolação é feita no espaço das coordenadas, não no espaço das permutações,
        o que permite uma transição suave entre as rotas.
        
        Parâmetros:
        t: Parâmetro da homotopia (0 <= t <= 1)
        """
        # Para t=0, retorna a rota inicial
        if t == 0:
            return self.rota_inicial
        
        # Para t=1, retorna a rota final
        if t == 1:
            return self.rota_final
        
        # Criar representação das rotas como pontos contínuos no espaço
        pontos_rota_inicial = np.array([self.cidades[i] for i in self.rota_inicial])
        pontos_rota_final = np.array([self.cidades[i] for i in self.rota_final])
        
        # Interpolar linearmente entre as duas rotas
        pontos_interpolados = (1-t) * pontos_rota_inicial + t * pontos_rota_final
        
        # Para valores intermediários, precisamos mapear de volta para as cidades discretas
        # Usamos o algoritmo do vizinho mais próximo para cada ponto interpolado
        rota_interpolada = []
        cidades_disponíveis = set(range(self.num_cidades))
        
        # Para cada ponto na interpolação, encontre a cidade mais próxima ainda não alocada
        for ponto in pontos_interpolados[:-1]:  # Excluímos o último ponto que é duplicado
            if not cidades_disponíveis:
                break
                
            # Calcular distâncias do ponto interpolado para todas as cidades disponíveis
            distancias = np.array([
                np.linalg.norm(ponto - self.cidades[i]) if i in cidades_disponíveis
                else float('inf') for i in range(self.num_cidades)
            ])
            
            # Encontrar a cidade mais próxima
            cidade_mais_próxima = np.argmin(distancias)
            rota_interpolada.append(cidade_mais_próxima)
            cidades_disponíveis.remove(cidade_mais_próxima)
        
        # Adicionar as cidades restantes não alocadas
        rota_interpolada.extend(cidades_disponíveis)
        
        # Retornar ao ponto inicial para completar o ciclo
        rota_interpolada.append(rota_interpolada[0])
        
        return rota_interpolada

    def visualizar_homotopia(self, num_etapas=50, salvar_gif=False):
        """
        Visualiza a homotopia entre a rota inicial e a rota final.
        
        Parâmetros:
        num_etapas: Número de etapas intermediárias na homotopia
        salvar_gif: Se True, salva a animação como GIF
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotar as cidades
        ax.scatter(self.cidades[:, 0], self.cidades[:, 1], c='blue', s=100, zorder=2)
        
        # Adicionar rótulos às cidades
        for i, (x, y) in enumerate(self.cidades):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Configurações do gráfico
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Homotopia de Caminhos no Problema do Caixeiro Viajante')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        
        # Preparar linha para a rota
        linha, = ax.plot([], [], 'r-', lw=2, zorder=1)
        info_texto = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Função de inicialização da animação
        def init():
            linha.set_data([], [])
            info_texto.set_text('')
            return linha, info_texto
        
        # Função de atualização para a animação
        def update(frame):
            t = frame / (num_etapas - 1) if num_etapas > 1 else 0
            rota = self.interpolar_rota(t)
            
            # Obter coordenadas para a rota
            x_rota = [self.cidades[i][0] for i in rota]
            y_rota = [self.cidades[i][1] for i in rota]
            
            linha.set_data(x_rota, y_rota)
            
            # Calcular comprimento da rota
            comprimento = self.calcular_comprimento_rota(rota)
            info_texto.set_text(f't = {t:.2f}\nComprimento: {comprimento:.2f}')
            
            return linha, info_texto
        
        # Criar animação
        ani = FuncAnimation(fig, update, frames=num_etapas, init_func=init, 
                           blit=True, interval=200)
        
        # Salvar animação se solicitado
        if salvar_gif:
            ani.save('tsp_homotopia.gif', writer='pillow', fps=10)
        
        plt.tight_layout()
        plt.show()
        
        # Visualizar também as rotas inicial e final para comparação
        self.visualizar_rotas_comparativas()
    
    def visualizar_rotas_comparativas(self):
        """Visualiza lado a lado as rotas inicial e final para comparação."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plotar rota inicial
        self._plotar_rota(ax1, self.rota_inicial, "Rota Inicial Aleatória")
        
        # Plotar rota final
        self._plotar_rota(ax2, self.rota_final, "Rota Final Otimizada")
        
        plt.tight_layout()
        plt.show()
    
    def _plotar_rota(self, ax, rota, titulo):
        """Função auxiliar para plotar uma rota específica."""
        # Plotar cidades
        ax.scatter(self.cidades[:, 0], self.cidades[:, 1], c='blue', s=100, zorder=2)
        
        # Adicionar rótulos
        for i, (x, y) in enumerate(self.cidades):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plotar rota
        x_rota = [self.cidades[i][0] for i in rota]
        y_rota = [self.cidades[i][1] for i in rota]
        ax.plot(x_rota, y_rota, 'r-', lw=2, zorder=1)
        
        # Calcular comprimento
        comprimento = self.calcular_comprimento_rota(rota)
        
        # Configurações do gráfico
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{titulo}\nComprimento: {comprimento:.2f}")
        ax.grid(True)

# Exemplo de uso
if __name__ == "__main__":
    # Criar instância com 15 cidades
    tsp = TSPHomotopia(num_cidades=15, seed=42)
    
    # Visualizar a homotopia
    tsp.visualizar_homotopia(num_etapas=30, salvar_gif=False)