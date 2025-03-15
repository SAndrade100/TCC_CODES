import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import networkx as nx
import random
from scipy.interpolate import splprep, splev
import time
from collections import defaultdict
import heapq

class TSPHomotopiaMapa:
    def __init__(self, pontos=None, conexoes=None, seed=42):
        """
        Inicializa o problema do caixeiro viajante com homotopia para um mapa.
        
        Parâmetros:
        pontos: Lista de coordenadas (x, y) dos pontos no mapa
                Se None, gera pontos aleatórios
        conexoes: Lista de tuplas (i, j, distancia) indicando as conexões
                  Se None, cria uma malha de conexões
        seed: Semente para reprodutibilidade
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Configurar pontos (cidades/nós)
        if pontos is None:
            # Gerar pontos de exemplo
            self.num_pontos = 15
            self.pontos = np.random.rand(self.num_pontos, 2)
        else:
            self.pontos = np.array(pontos)
            self.num_pontos = len(self.pontos)
        
        print(f"Inicializando TSP com {self.num_pontos} pontos no mapa...")
        
        # Criar grafo
        self.grafo = nx.Graph()
        for i in range(self.num_pontos):
            self.grafo.add_node(i, pos=tuple(self.pontos[i]))
        
        # Adicionar conexões/rotas
        if conexoes is None:
            # Se não forem fornecidas conexões, criar um grafo conectado
            # Usando uma combinação de conexões próximas e algumas aleatórias
            self._criar_conexoes_exemplo()
        else:
            for i, j, dist in conexoes:
                self.grafo.add_edge(i, j, weight=dist)
        
        # Verificar se o grafo é conectado
        if not nx.is_connected(self.grafo):
            print("AVISO: O grafo não é conectado. Adicionando conexões para garantir conectividade.")
            self._garantir_conectividade()
        
        # Calcular matriz de distâncias com base nos caminhos mais curtos
        print("Calculando matriz de distâncias baseada no mapa...")
        self.calcular_matriz_distancias_grafo()
        
        # Resolver o TSP usando um algoritmo aproximado
        print("Resolvendo o problema do caixeiro viajante...")
        self.solucao_tsp = self.resolver_tsp_aproximado()
        
        # Criar o caminho da solução
        self.caminho_tsp = self.criar_caminho_completo(self.solucao_tsp)
        
        # Criar objeto convexo ao redor dos pontos
        print("Criando objeto convexo inicial...")
        self.criar_objeto_convexo()
        
        # Cache para armazenar resultados da homotopia
        self.cache_homotopia = {}
    
    def _criar_conexoes_exemplo(self):
        """Cria conexões de exemplo para o grafo."""
        # Conectar cada ponto aos seus 3 vizinhos mais próximos
        for i in range(self.num_pontos):
            distancias = [(j, np.linalg.norm(self.pontos[i] - self.pontos[j]))
                         for j in range(self.num_pontos) if i != j]
            distancias_ordenadas = sorted(distancias, key=lambda x: x[1])
            
            # Conectar aos 3 mais próximos (ou menos se não houver 3)
            num_vizinhos = min(3, len(distancias_ordenadas))
            for j in range(num_vizinhos):
                vizinho, dist = distancias_ordenadas[j]
                self.grafo.add_edge(i, vizinho, weight=dist)
        
        # Adicionar algumas conexões aleatórias para diversificar
        for _ in range(self.num_pontos // 2):
            i = random.randint(0, self.num_pontos - 1)
            j = random.randint(0, self.num_pontos - 1)
            if i != j and not self.grafo.has_edge(i, j):
                dist = np.linalg.norm(self.pontos[i] - self.pontos[j])
                self.grafo.add_edge(i, j, weight=dist)
    
    def _garantir_conectividade(self):
        """Adiciona arestas para garantir que o grafo seja conectado."""
        # Identificar os componentes conectados
        componentes = list(nx.connected_components(self.grafo))
        
        # Se já estiver conectado, não fazer nada
        if len(componentes) == 1:
            return
        
        # Conectar cada componente ao próximo
        for i in range(len(componentes) - 1):
            comp1 = list(componentes[i])
            comp2 = list(componentes[i + 1])
            
            # Encontrar os nós mais próximos entre os componentes
            melhor_dist = float('inf')
            melhor_par = None
            
            for n1 in comp1:
                for n2 in comp2:
                    dist = np.linalg.norm(self.pontos[n1] - self.pontos[n2])
                    if dist < melhor_dist:
                        melhor_dist = dist
                        melhor_par = (n1, n2)
            
            # Adicionar a aresta conectando os componentes
            if melhor_par:
                n1, n2 = melhor_par
                self.grafo.add_edge(n1, n2, weight=melhor_dist)
    
    def calcular_matriz_distancias_grafo(self):
        """
        Calcula a matriz de distâncias com base nos caminhos mais curtos no grafo.
        Esta é a diferença fundamental: as distâncias são baseadas nas rotas disponíveis,
        não em distâncias euclidianas diretas.
        """
        # Computar todos os caminhos mais curtos usando o algoritmo de Floyd-Warshall
        caminhos = dict(nx.all_pairs_dijkstra_path(self.grafo, weight='weight'))
        comprimentos = dict(nx.all_pairs_dijkstra_path_length(self.grafo, weight='weight'))
        
        # Armazenar os caminhos e comprimentos
        self.caminhos_mais_curtos = caminhos
        
        # Criar a matriz de distâncias
        self.matriz_distancias = np.zeros((self.num_pontos, self.num_pontos))
        for i in range(self.num_pontos):
            for j in range(self.num_pontos):
                if i == j:
                    self.matriz_distancias[i, j] = 0
                else:
                    self.matriz_distancias[i, j] = comprimentos[i][j]
    
    def criar_caminho_completo(self, rota):
        """
        Cria um caminho completo passando por todas as arestas reais do grafo
        para a rota da solução do TSP.
        """
        caminho_completo = []
        
        # Para cada par de cidades consecutivas na rota
        for i in range(len(rota)):
            cidade_atual = rota[i]
            proxima_cidade = rota[(i + 1) % len(rota)]
            
            # Obter o caminho mais curto entre estas cidades
            caminho_parcial = self.caminhos_mais_curtos[cidade_atual][proxima_cidade]
            
            # Adicionar todas as cidades neste caminho, exceto a última (será o início do próximo)
            caminho_completo.extend(caminho_parcial[:-1])
        
        # Adicionar a primeira cidade para fechar o ciclo
        caminho_completo.append(caminho_completo[0])
        
        # Converter para coordenadas
        coordenadas_caminho = np.array([self.pontos[i] for i in caminho_completo])
        
        return coordenadas_caminho
    
    def resolver_tsp_aproximado(self):
        """
        Resolve o TSP usando uma heurística adaptada para grafos.
        """
        # Usar uma combinação de nearest neighbor e 2-opt
        
        # 1. Nearest Neighbor com múltiplos pontos de partida
        melhor_rota = None
        melhor_custo = float('inf')
        
        # Tentar diferentes pontos de partida
        for inicio in range(min(10, self.num_pontos)):
            atual = inicio
            rota = [atual]
            nao_visitados = set(range(self.num_pontos)) - {atual}
            
            while nao_visitados:
                proximo = min(nao_visitados, 
                             key=lambda x: self.matriz_distancias[atual][x])
                rota.append(proximo)
                nao_visitados.remove(proximo)
                atual = proximo
            
            # Calcular o custo total da rota
            custo = sum(self.matriz_distancias[rota[i]][rota[(i+1) % len(rota)]] 
                        for i in range(len(rota)))
            
            if custo < melhor_custo:
                melhor_custo = custo
                melhor_rota = rota
        
        # 2. Aplicar 2-opt para melhorar a rota
        melhorado = True
        iteracoes = 0
        max_iteracoes = min(100, self.num_pontos * 5)
        
        while melhorado and iteracoes < max_iteracoes:
            melhorado = False
            iteracoes += 1
            
            for i in range(1, len(melhor_rota) - 2):
                for j in range(i + 1, len(melhor_rota)):
                    # Calcular o ganho da troca 2-opt
                    if j - i == 1:
                        continue  # Pular arestas adjacentes
                    
                    # Cidades envolvidas
                    a, b = melhor_rota[i-1], melhor_rota[i]
                    c, d = melhor_rota[j], melhor_rota[(j+1) % len(melhor_rota)]
                    
                    # Calcular custo atual e novo
                    custo_atual = self.matriz_distancias[a][b] + self.matriz_distancias[c][d]
                    novo_custo = self.matriz_distancias[a][c] + self.matriz_distancias[b][d]
                    
                    if novo_custo < custo_atual:
                        # Aplicar a troca 2-opt
                        melhor_rota[i:j+1] = melhor_rota[i:j+1][::-1]
                        melhorado = True
                        break
                
                if melhorado:
                    break
        
        return melhor_rota
    
    def criar_objeto_convexo(self, tipo='fecho_suave', num_pontos=100):
        """
        Cria o objeto convexo inicial para a homotopia.
        """
        # Encontrar o centro dos pontos
        centro = np.mean(self.pontos, axis=0)
        
        if tipo == 'circulo':
            # Criar um círculo ao redor dos pontos
            raio = max(np.linalg.norm(self.pontos - centro, axis=1)) * 1.1
            theta = np.linspace(0, 2*np.pi, num_pontos, endpoint=True)
            self.objeto_convexo = np.column_stack((
                centro[0] + raio * np.cos(theta),
                centro[1] + raio * np.sin(theta)
            ))
            
        elif tipo == 'fecho_suave':
            try:
                # Calcular o fecho convexo dos pontos
                hull = ConvexHull(self.pontos)
                vertices_hull = self.pontos[hull.vertices]
                
                # Fechar o contorno repetindo o primeiro vértice
                contorno_fechado = np.vstack([vertices_hull, vertices_hull[0]])
                
                # Suavizar o contorno usando splines
                tck, u = splprep([contorno_fechado[:, 0], contorno_fechado[:, 1]], s=0, per=True)
                u_new = np.linspace(0, 1, num_pontos)
                suave = np.column_stack(splev(u_new, tck))
                
                # Expandir ligeiramente o contorno
                self.objeto_convexo = centro + (suave - centro) * 1.1
            except:
                # Fallback para círculo se o fecho convexo falhar
                print("Aviso: Falha ao criar fecho convexo suave. Usando círculo.")
                raio = max(np.linalg.norm(self.pontos - centro, axis=1)) * 1.1
                theta = np.linspace(0, 2*np.pi, num_pontos, endpoint=True)
                self.objeto_convexo = np.column_stack((
                    centro[0] + raio * np.cos(theta),
                    centro[1] + raio * np.sin(theta)
                ))
        else:
            raise ValueError("Tipo de objeto convexo desconhecido. Use 'circulo' ou 'fecho_suave'")
    
    def interpolar_homotopia(self, t):
        """
        Interpola entre o objeto convexo e o caminho do TSP.
        
        Parâmetros:
        t: Parâmetro da homotopia (0 <= t <= 1)
        """
        # Verificar se já está no cache
        if t in self.cache_homotopia:
            return self.cache_homotopia[t]
        
        if t == 0:
            resultado = self.objeto_convexo
        elif t == 1:
            resultado = self.caminho_tsp
        else:
            # Reamostrar os pontos para garantir correspondência
            n_pontos = len(self.objeto_convexo)
            tsp_resampled = np.zeros((n_pontos, 2))
            
            indices = np.linspace(0, len(self.caminho_tsp) - 1, n_pontos, endpoint=True)
            indices_int = indices.astype(int)
            frac = indices - indices_int
            
            # Reamostrar o caminho TSP
            for i in range(n_pontos):
                if i == n_pontos - 1 or indices_int[i] == len(self.caminho_tsp) - 1:
                    tsp_resampled[i] = self.caminho_tsp[indices_int[i]]
                else:
                    # Interpolação linear entre pontos consecutivos
                    p1 = self.caminho_tsp[indices_int[i]]
                    p2 = self.caminho_tsp[indices_int[i] + 1]
                    tsp_resampled[i] = p1 + frac[i] * (p2 - p1)
            
            # Interpolação linear entre o objeto convexo e o caminho TSP reamostrado
            resultado = (1 - t) * self.objeto_convexo + t * tsp_resampled
        
        # Armazenar no cache para reutilização
        self.cache_homotopia[t] = resultado
        return resultado

    def animar_homotopia(self, num_frames=100, intervalo=50):
        """
        Cria uma animação da homotopia entre o objeto convexo e o caminho do TSP.
        
        Parâmetros:
        num_frames: Número de frames na animação
        intervalo: Intervalo entre frames em milissegundos
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.set_title("Homotopia entre Objeto Convexo e Caminho TSP")
        
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
            pontos.set_data(self.pontos[:, 0], self.pontos[:, 1])
            
            return linha, pontos
        
        # Criar a animação
        anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=intervalo, blit=True)
        
        plt.close()
        return anim

    def plotar_resultados(self):
        """
        Plota o resultado final da homotopia, mostrando o objeto convexo,
        o caminho do TSP e os pontos.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.set_title("Resultado Final da Homotopia")
        
        # Plotar o objeto convexo
        ax.plot(self.objeto_convexo[:, 0], self.objeto_convexo[:, 1], 'g--', lw=2, label="Objeto Convexo")
        
        # Plotar o caminho do TSP
        ax.plot(self.caminho_tsp[:, 0], self.caminho_tsp[:, 1], 'b-', lw=2, label="Caminho TSP")
        
        # Plotar os pontos
        ax.plot(self.pontos[:, 0], self.pontos[:, 1], 'ro', markersize=5, label="Pontos")
        
        ax.legend()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Criar uma instância do TSP com um mapa personalizado
    pontos = np.random.rand(15, 2)  # 15 pontos aleatórios
    tsp = TSPHomotopiaMapa(pontos=pontos)
    
    # Gerar e exibir a animação da homotopia
    anim = tsp.animar_homotopia(num_frames=100, intervalo=50)
    plt.show()
    
    # Plotar o resultado final
    tsp.plotar_resultados()