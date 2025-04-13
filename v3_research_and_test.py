import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import networkx as nx
import random
from scipy.interpolate import splprep, splev
import time
from tqdm import tqdm
import matplotlib as mpl

class TSPHomotopiaAvancada:
    def __init__(self, num_cidades=50, seed=42, algoritmo='nearest_neighbor'):
        """
        Inicializa o problema do caixeiro viajante com homotopia convexa.
        Versão otimizada para lidar com um número maior de cidades.
        
        Parâmetros:
        num_cidades: Número de cidades
        seed: Semente para reprodutibilidade
        algoritmo: Algoritmo para resolver o TSP ('nearest_neighbor', '2opt', 'christofides')
        """
        self.num_cidades = num_cidades
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Gerando problema TSP com {num_cidades} cidades...")
        
        # Gerar cidades aleatórias em um plano 2D
        self.cidades = np.random.rand(num_cidades, 2)
        
        # Calcular o fecho convexo das cidades
        hull = ConvexHull(self.cidades)
        self.vertices_convexos = hull.vertices
        
        # Calcular matriz de distâncias de forma eficiente
        print("Calculando matriz de distâncias...")
        self.calcular_matriz_distancias()
        
        # Resolver o TSP usando o algoritmo escolhido
        print(f"Resolvendo TSP usando algoritmo {algoritmo}...")
        if algoritmo == 'nearest_neighbor':
            self.solucao_tsp = self.resolver_tsp_nn()
        elif algoritmo == '2opt':
            self.solucao_tsp = self.resolver_tsp_2opt()
        elif algoritmo == 'christofides':
            # Implementação simplificada do algoritmo de Christofides
            self.solucao_tsp = self.resolver_tsp_aproximado()
        else:
            raise ValueError("Algoritmo desconhecido. Escolha 'nearest_neighbor', '2opt' ou 'christofides'")
        
        # Obter coordenadas das cidades na ordem da solução TSP
        self.coordenadas_tsp = np.array([self.cidades[i] for i in self.solucao_tsp + [self.solucao_tsp[0]]])
        
        # Criar objeto convexo ao redor das cidades
        print("Criando objeto convexo inicial...")
        self.criar_objeto_convexo()
        
        # Pré-calcular as etapas da homotopia
        self.cache_homotopia = {}
    
    def calcular_matriz_distancias(self):
        """Calcula a matriz de distâncias entre todas as cidades de forma eficiente."""
        self.matriz_distancias = np.zeros((self.num_cidades, self.num_cidades))
        for i in range(self.num_cidades):
            # Aproveitando a simetria para economizar cálculos
            for j in range(i + 1, self.num_cidades):
                dist = np.linalg.norm(self.cidades[i] - self.cidades[j])
                self.matriz_distancias[i, j] = dist
                self.matriz_distancias[j, i] = dist
    
    def criar_objeto_convexo(self, tipo_objeto='fecho_suave', num_pontos=200):
        """
        Cria um objeto convexo ao redor das cidades.
        
        Parâmetros:
        tipo_objeto: Tipo de objeto convexo ('circulo', 'fecho_suave', 'elipse')
        num_pontos: Número de pontos para representar o objeto
        """
        # Encontrar o centro das cidades
        centro = np.mean(self.cidades, axis=0)
        
        # Calcular dimensões
        min_coords = np.min(self.cidades, axis=0)
        max_coords = np.max(self.cidades, axis=0)
        largura = max_coords[0] - min_coords[0]
        altura = max_coords[1] - min_coords[1]
        
        if tipo_objeto == 'circulo':
            # Criar um círculo como objeto convexo
            raio = max(np.linalg.norm(self.cidades - centro, axis=1)) * 1.1
            theta = np.linspace(0, 2*np.pi, num_pontos, endpoint=True)
            self.circulo_x = centro[0] + raio * np.cos(theta)
            self.circulo_y = centro[1] + raio * np.sin(theta)
            self.objeto_convexo = np.column_stack((self.circulo_x, self.circulo_y))
            
        elif tipo_objeto == 'elipse':
            # Criar uma elipse como objeto convexo
            a = largura * 0.6  # Semi-eixo maior
            b = altura * 0.6   # Semi-eixo menor
            theta = np.linspace(0, 2*np.pi, num_pontos, endpoint=True)
            self.objeto_convexo = np.column_stack((
                centro[0] + a * np.cos(theta),
                centro[1] + b * np.sin(theta)
            ))
            
        elif tipo_objeto == 'fecho_suave':
            # Usar o fecho convexo e suavizá-lo
            hull_points = self.cidades[self.vertices_convexos]
            # Adicionar o primeiro ponto novamente para fechar o ciclo
            hull_closed = np.vstack([hull_points, hull_points[0]])
            
            # Suavizar o fecho convexo com uma spline
            try:
                tck, u = splprep([hull_closed[:, 0], hull_closed[:, 1]], s=0, per=True)
                u_new = np.linspace(0, 1, num_pontos)
                smooth_hull = np.column_stack(splev(u_new, tck))
                # Expandir ligeiramente para englobar todas as cidades
                expanded_hull = centro + (smooth_hull - centro) * 1.05
                self.objeto_convexo = expanded_hull
            except:
                # Fallback para círculo se a spline falhar
                print("Aviso: Falha ao suavizar o fecho convexo. Usando círculo como fallback.")
                raio = max(np.linalg.norm(self.cidades - centro, axis=1)) * 1.1
                theta = np.linspace(0, 2*np.pi, num_pontos, endpoint=True)
                self.objeto_convexo = np.column_stack((
                    centro[0] + raio * np.cos(theta),
                    centro[1] + raio * np.sin(theta)
                ))
        else:
            raise ValueError("Tipo de objeto desconhecido. Use 'circulo', 'fecho_suave' ou 'elipse'")

    def resolver_tsp_nn(self):
        """Resolve o TSP usando o algoritmo do vizinho mais próximo."""
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
    
    def resolver_tsp_2opt(self):
        """
        Resolve o TSP usando o algoritmo 2-opt para melhorar uma rota inicial.
        Começa com a solução do vizinho mais próximo e aplica otimização 2-opt.
        """
        # Obter uma solução inicial usando o algoritmo do vizinho mais próximo
        rota = self.resolver_tsp_nn()
        
        melhorado = True
        n_iteracoes = 0
        max_iteracoes = min(100, self.num_cidades * 5)  # Limitar número de iterações
        
        while melhorado and n_iteracoes < max_iteracoes:
            melhorado = False
            n_iteracoes += 1
            
            # Tentar todas as trocas 2-opt possíveis
            for i in range(1, len(rota) - 2):
                for j in range(i + 1, len(rota)):
                    if j - i == 1:
                        continue  # Pular arestas adjacentes
                    
                    # Calcular a mudança no comprimento da rota se invertermos o caminho de i a j
                    # Aresta atual: (i-1, i) e (j, j+1 ou 0)
                    # Nova aresta: (i-1, j) e (i, j+1 ou 0)
                    
                    # Cidade atual após i-1 é i
                    atual_anterior = rota[i-1]
                    atual_i = rota[i]
                    # Cidade atual após j é j+1 ou 0 (caso j seja o último)
                    atual_j = rota[j]
                    atual_proximo = rota[(j+1) % len(rota)]
                    
                    # Calcular custo atual
                    custo_atual = (self.matriz_distancias[atual_anterior, atual_i] + 
                                 self.matriz_distancias[atual_j, atual_proximo])
                    
                    # Calcular novo custo após troca
                    novo_custo = (self.matriz_distancias[atual_anterior, atual_j] + 
                                 self.matriz_distancias[atual_i, atual_proximo])
                    
                    # Se o novo caminho for melhor, fazer a troca 2-opt
                    if novo_custo < custo_atual:
                        # Inverter o segmento entre i e j
                        rota[i:j+1] = rota[i:j+1][::-1]
                        melhorado = True
                        break
                
                if melhorado:
                    break
        
        return rota
    
    def resolver_tsp_aproximado(self):
        """
        Implementação simplificada de um algoritmo aproximado para o TSP.
        Usa uma combinação de árvore geradora mínima e correspondência mínima.
        """
        # Este é um placeholder para um algoritmo mais avançado
        # Por ora, vamos usar uma versão melhorada do vizinho mais próximo
        
        # Implementar múltiplas execuções do vizinho mais próximo
        # com diferentes pontos de partida e escolher o melhor
        melhor_rota = None
        melhor_comprimento = float('inf')
        
        # Tentar alguns pontos de partida diferentes
        num_tentativas = min(10, self.num_cidades)
        pontos_partida = random.sample(range(self.num_cidades), num_tentativas)
        
        for ponto in pontos_partida:
            # Algoritmo do vizinho mais próximo a partir deste ponto
            cidade_atual = ponto
            rota = [cidade_atual]
            cidades_nao_visitadas = set(range(self.num_cidades)) - {cidade_atual}
            
            while cidades_nao_visitadas:
                proxima_cidade = min(cidades_nao_visitadas, 
                                  key=lambda cidade: self.matriz_distancias[cidade_atual][cidade])
                rota.append(proxima_cidade)
                cidades_nao_visitadas.remove(proxima_cidade)
                cidade_atual = proxima_cidade
            
            # Calcular comprimento desta rota
            comprimento = self.calcular_comprimento_rota(rota + [rota[0]])
            
            # Atualizar se for melhor
            if comprimento < melhor_comprimento:
                melhor_comprimento = comprimento
                melhor_rota = rota
        
        # Aplicar otimização 2-opt à melhor rota
        melhorado = True
        n_iteracoes = 0
        max_iteracoes = min(50, self.num_cidades * 3)  # Limitar número de iterações
        
        while melhorado and n_iteracoes < max_iteracoes:
            melhorado = False
            n_iteracoes += 1
            
            # Tentar um número limitado de trocas aleatórias
            max_tentativas = min(100, self.num_cidades * 10)
            for _ in range(max_tentativas):
                # Escolher dois índices aleatórios i < j
                i = random.randint(1, len(melhor_rota) - 3)
                j = random.randint(i + 1, len(melhor_rota) - 1)
                
                # Calcular a mudança no comprimento se invertermos o caminho de i a j
                atual_anterior = melhor_rota[i-1]
                atual_i = melhor_rota[i]
                atual_j = melhor_rota[j]
                atual_proximo = melhor_rota[(j+1) % len(melhor_rota)]
                
                custo_atual = (self.matriz_distancias[atual_anterior, atual_i] + 
                             self.matriz_distancias[atual_j, atual_proximo])
                novo_custo = (self.matriz_distancias[atual_anterior, atual_j] + 
                             self.matriz_distancias[atual_i, atual_proximo])
                
                if novo_custo < custo_atual:
                    # Inverter o segmento
                    melhor_rota[i:j+1] = melhor_rota[i:j+1][::-1]
                    melhorado = True
                    break
        
        return melhor_rota
    
    def calcular_comprimento_rota(self, rota):
        """Calcula o comprimento total de uma rota."""
        comprimento = 0
        for i in range(len(rota)-1):
            comprimento += self.matriz_distancias[rota[i], rota[i+1]]
        return comprimento
    
    def interpolar_homotopia(self, t, cache=True):
        """
        Interpola entre o objeto convexo e a solução do TSP.
        
        Parâmetros:
        t: Parâmetro da homotopia (0 <= t <= 1)
        cache: Se True, armazena os resultados em cache para reutilização
        """
        # Verificar se já está no cache
        if cache and t in self.cache_homotopia:
            return self.cache_homotopia[t]
        
        if t == 0:
            resultado = self.objeto_convexo
        elif t == 1:
            resultado = self.coordenadas_tsp
        else:
            # Interpolação linear entre o objeto convexo e a solução do TSP
            resultado = (1 - t) * self.objeto_convexo + t * self.coordenadas_tsp
        
        # Armazenar no cache se necessário
        if cache:
            self.cache_homotopia[t] = resultado
        
        return resultado

    def animar_homotopia(self, num_frames=100, intervalo=50):
        """
        Cria uma animação da homotopia entre o objeto convexo e a solução do TSP.
        
        Parâmetros:
        num_frames: Número de frames na animação
        intervalo: Intervalo entre frames em milissegundos
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.set_title("Homotopia entre Objeto Convexo e Solução TSP")
        
        # Inicializar o plot
        linha, = ax.plot([], [], 'b-', lw=2)
        pontos, = ax.plot([], [], 'ro', markersize=5)
        
        def init():
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return linha, pontos
        
        def update(frame):
            t = frame / (num_frames - 1)
            pontos_interpolados = self.interpolar_homotopia(t)
            
            # Atualizar a linha e os pontos
            linha.set_data(pontos_interpolados[:, 0], pontos_interpolados[:, 1])
            pontos.set_data(self.cidades[:, 0], self.cidades[:, 1])
            
            return linha, pontos
        
        # Criar a animação
        anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=intervalo, blit=True)
        
        plt.close()
        return anim

    def plotar_resultados(self):
        """
        Plota o resultado final da homotopia, mostrando o objeto convexo,
        a solução do TSP e as cidades.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.set_title("Resultado Final da Homotopia")
        
        # Plotar o objeto convexo
        ax.plot(self.objeto_convexo[:, 0], self.objeto_convexo[:, 1], 'g--', lw=2, label="Objeto Convexo")
        
        # Plotar a solução do TSP
        ax.plot(self.coordenadas_tsp[:, 0], self.coordenadas_tsp[:, 1], 'b-', lw=2, label="Solução TSP")
        
        # Plotar as cidades
        ax.plot(self.cidades[:, 0], self.cidades[:, 1], 'ro', markersize=5, label="Cidades")
        
        ax.legend()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    tsp = TSPHomotopiaAvancada(num_cidades=30, algoritmo='2opt')
    anim = tsp.animar_homotopia(num_frames=100, intervalo=50)
    plt.show()
    tsp.plotar_resultados()
