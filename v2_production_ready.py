import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import networkx as nx
import random
from scipy.interpolate import splprep, splev

class TSPHomotopiaConvexa:
    def __init__(self, num_cidades=15, seed=42):
        """
        Inicializa o problema do caixeiro viajante com homotopia convexa.
        
        Parâmetros:
        num_cidades: Número de cidades
        seed: Semente para reprodutibilidade
        """
        self.num_cidades = num_cidades
        np.random.seed(seed)
        random.seed(seed)
        
        # Gerar cidades aleatórias em um plano 2D
        self.cidades = np.random.rand(num_cidades, 2)
        
        # Calcular o fecho convexo das cidades
        hull = ConvexHull(self.cidades)
        self.vertices_convexos = hull.vertices
        
        # Calcular matriz de distâncias
        self.matriz_distancias = np.zeros((num_cidades, num_cidades))
        for i in range(num_cidades):
            for j in range(num_cidades):
                self.matriz_distancias[i, j] = np.linalg.norm(self.cidades[i] - self.cidades[j])
        
        # Resolver o TSP usando o algoritmo do vizinho mais próximo
        self.solucao_tsp = self.resolver_tsp()
        
        # Obter coordenadas das cidades na ordem da solução TSP
        self.coordenadas_tsp = np.array([self.cidades[i] for i in self.solucao_tsp + [self.solucao_tsp[0]]])
        
        # Criar objeto convexo ao redor das cidades
        self.criar_objeto_convexo()
    
    def criar_objeto_convexo(self):
        """Cria um objeto convexo ao redor das cidades (círculo ou polígono convexo)."""
        # Encontrar o centro das cidades
        centro = np.mean(self.cidades, axis=0)
        
        # Encontrar o raio que envolve todas as cidades
        raio = max(np.linalg.norm(self.cidades - centro, axis=1)) * 1.1
        
        # Criar um círculo como objeto convexo
        theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
        self.circulo_x = centro[0] + raio * np.cos(theta)
        self.circulo_y = centro[1] + raio * np.sin(theta)
        self.objeto_convexo = np.column_stack((self.circulo_x, self.circulo_y))
        
        # Alternativa: usar o fecho convexo das cidades como objeto inicial
        # (descomente as linhas abaixo para usar essa alternativa)
        """
        hull_points = self.cidades[self.vertices_convexos]
        # Adicionar o primeiro ponto novamente para fechar o ciclo
        hull_closed = np.vstack([hull_points, hull_points[0]])
        
        # Suavizar o fecho convexo com uma spline
        tck, u = splprep([hull_closed[:, 0], hull_closed[:, 1]], s=0, per=True)
        u_new = np.linspace(0, 1, 100)
        smooth_hull = np.column_stack(splev(u_new, tck))
        self.objeto_convexo = smooth_hull
        """

    def resolver_tsp(self):
        """Resolve o problema do caixeiro viajante usando o algoritmo do vizinho mais próximo."""
        # Começar de um ponto aleatório
        cidade_atual = random.randint(0, self.num_cidades - 1)
        rota = [cidade_atual]
        cidades_nao_visitadas = set(range(self.num_cidades)) - {cidade_atual}
        
        # Construir a rota
        while cidades_nao_visitadas:
            proxima_cidade = min(cidades_nao_visitadas, 
                              key=lambda cidade: self.matriz_distancias[cidade_atual][cidade])
            rota.append(proxima_cidade)
            cidades_nao_visitadas.remove(proxima_cidade)
            cidade_atual = proxima_cidade
        
        return rota
    
    def calcular_comprimento_rota(self, rota):
        """Calcula o comprimento total de uma rota."""
        comprimento = 0
        for i in range(len(rota)-1):
            comprimento += self.matriz_distancias[rota[i], rota[i+1]]
        return comprimento
    
    def interpolar_homotopia(self, t):
        """
        Interpola entre o objeto convexo e a solução do TSP.
        
        Parâmetros:
        t: Parâmetro da homotopia (0 <= t <= 1)
          t=0: objeto convexo original
          t=1: solução do TSP
        """
        if t == 0:
            return self.objeto_convexo
        elif t == 1:
            return self.coordenadas_tsp
        
        # Encontrar correspondência entre pontos do objeto convexo e da solução TSP
        n_pontos = len(self.objeto_convexo)
        n_tsp = len(self.coordenadas_tsp)
        
        # Reamostrar a solução TSP para ter o mesmo número de pontos que o objeto convexo
        indices = np.linspace(0, n_tsp - 1, n_pontos, endpoint=True)
        indices_int = indices.astype(int)
        frac = indices - indices_int
        
        tsp_resampled = np.zeros((n_pontos, 2))
        for i in range(n_pontos):
            if i == n_pontos - 1 or indices_int[i] == n_tsp - 1:
                tsp_resampled[i] = self.coordenadas_tsp[indices_int[i]]
            else:
                p1 = self.coordenadas_tsp[indices_int[i]]
                p2 = self.coordenadas_tsp[indices_int[i] + 1]
                tsp_resampled[i] = p1 + frac[i] * (p2 - p1)
        
        # Interpolação linear entre os pontos correspondentes
        pontos_interpolados = (1 - t) * self.objeto_convexo + t * tsp_resampled
                
        return pontos_interpolados
    
    def visualizar_homotopia(self, num_etapas=50, salvar_gif=False):
        """
        Visualiza a homotopia do objeto convexo para a solução do TSP.
        
        Parâmetros:
        num_etapas: Número de etapas na animação
        salvar_gif: Se True, salva a animação como GIF
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotar as cidades
        ax.scatter(self.cidades[:, 0], self.cidades[:, 1], c='blue', s=100, zorder=2)
        
        # Adicionar rótulos às cidades
        for i, (x, y) in enumerate(self.cidades):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Configurar os limites do gráfico
        border = 0.1
        ax.set_xlim(min(self.cidades[:, 0]) - border, max(self.cidades[:, 0]) + border)
        ax.set_ylim(min(self.cidades[:, 1]) - border, max(self.cidades[:, 1]) + border)
        
        # Títulos e rótulos
        ax.set_title('Homotopia de Objeto Convexo para Solução do TSP')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        
        # Linha para representar a curva atual
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
            pontos = self.interpolar_homotopia(t)
            linha.set_data(pontos[:, 0], pontos[:, 1])
            
            if t == 1:
                # Se for a solução final, calcular comprimento da rota TSP
                comprimento = self.calcular_comprimento_rota(self.solucao_tsp + [self.solucao_tsp[0]])
                info_texto.set_text(f't = {t:.2f}\nSolução TSP\nComprimento: {comprimento:.2f}')
            else:
                info_texto.set_text(f't = {t:.2f}\nDeformação em progresso')
            
            return linha, info_texto
        
        # Criar animação
        ani = FuncAnimation(fig, update, frames=num_etapas, init_func=init, 
                           blit=True, interval=200)
        
        # Salvar animação se solicitado
        if salvar_gif:
            ani.save('tsp_homotopia_convexa.gif', writer='pillow', fps=10)
        
        plt.tight_layout()
        plt.show()
    
    def visualizar_etapas_chave(self, num_etapas=5):
        """Visualiza algumas etapas-chave da homotopia lado a lado."""
        fig, axs = plt.subplots(1, num_etapas, figsize=(num_etapas*4, 4))
        
        for i, ax in enumerate(axs):
            t = i / (num_etapas - 1)
            pontos = self.interpolar_homotopia(t)
            
            # Plotar cidades
            ax.scatter(self.cidades[:, 0], self.cidades[:, 1], c='blue', s=50, zorder=2)
            
            # Plotar caminho atual
            ax.plot(pontos[:, 0], pontos[:, 1], 'r-', lw=2, zorder=1)
            
            # Configurar
            border = 0.1
            ax.set_xlim(min(self.cidades[:, 0]) - border, max(self.cidades[:, 0]) + border)
            ax.set_ylim(min(self.cidades[:, 1]) - border, max(self.cidades[:, 1]) + border)
            ax.set_title(f't = {t:.2f}')
            ax.grid(True)
            
            # Remover rótulos dos eixos exceto para o primeiro gráfico
            if i > 0:
                ax.set_yticklabels([])
            
        plt.tight_layout()
        plt.show()

    def aplicar_força_direcional(self):
        """
        Implementação avançada: usar forças direcionais para guiar a deformação.
        Esta é uma versão simplificada do algoritmo force-directed para ajudar
        na transição mais suave do objeto convexo para a solução TSP.
        """
        # Esta é uma extensão avançada que pode ser implementada para
        # melhorar a qualidade da homotopia. A ideia seria usar um algoritmo
        # de força direcional para guiar a deformação de maneira mais natural.
        pass

# Exemplo de uso
if __name__ == "__main__":
    # Criar uma instância do problema
    tsp = TSPHomotopiaConvexa(num_cidades=15, seed=42)
    
    # Visualizar a homotopia
    tsp.visualizar_homotopia(num_etapas=50, salvar_gif=False)
    
    # Visualizar algumas etapas-chave
    tsp.visualizar_etapas_chave(num_etapas=5)