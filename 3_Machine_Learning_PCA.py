import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==========================================
# 1. A Física (Motor Vetorizado - Super Rápido)
# ==========================================
def metropolis_vectorized(grid, beta):
    N = grid.shape[0]
    x, y = np.indices((N, N))
    
    for mask in [(x + y) % 2 == 0, (x + y) % 2 != 0]:
        vizinhos = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                    np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
        
        dE = 2 * grid * vizinhos
        prob = np.exp(-dE * beta)
        
        aceitar = (dE <= 0) | (np.random.rand(N, N) < prob)
        grid[mask & aceitar] *= -1
        
    return grid

# ==========================================
# 2. Coleta de Dados (Criando o Dataset para a IA)
# ==========================================
print("Gerando milhares de matrizes para a IA treinar...")

N = 20 # Matriz 20x20
temperaturas = np.linspace(1.0, 4.0, 40) # 40 temperaturas diferentes
X = [] # Aqui guardaremos as "fotos"
temperaturas_usadas = []

for T in temperaturas:
    beta = 1.0 / T
    grid = np.random.choice([-1, 1], size=(N, N))
    
    # 1. Termalização (Estabiliza o sistema)
    for _ in range(100): 
        grid = metropolis_vectorized(grid, beta)
        
    # 2. Coleta de amostras (Tira 50 fotos de cada temperatura)
    for _ in range(50):
        grid = metropolis_vectorized(grid, beta)
        X.append(grid.flatten()) # Achata a matriz 20x20 em 400 pixels
        temperaturas_usadas.append(T)

X = np.array(X)
temperaturas_usadas = np.array(temperaturas_usadas)
print(f"Dataset pronto! Entregando {X.shape[0]} matrizes para a IA analisar.")

# ==========================================
# 3. O Machine Learning (O Algoritmo PCA)
# ==========================================
print("Iniciando aprendizado da máquina (PCA)...")

# A IA vai resumir as 400 dimensões para apenas 1 (a regra oculta do sistema)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# ==========================================
# 4. O Gráfico da Descoberta
# ==========================================
plt.figure(figsize=(10, 6))

# Plotamos o que a IA descobriu contra a Temperatura real
plt.scatter(temperaturas_usadas, np.abs(X_pca), color='royalblue', alpha=0.6, edgecolor='k')

# Marcamos onde a física diz que o fenômeno deveria acontecer
plt.axvline(x=2.269, color='crimson', linestyle='--', linewidth=2, label='Temperatura Crítica Teórica ($T_c = 2.269$)')

plt.title("Descoberta da Transição de Fase via PCA Não Supervisionado", fontsize=14)
plt.xlabel("Temperatura do Sistema ($T$)", fontsize=12)
plt.ylabel("Componente Principal (A Ordem descoberta pela IA)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

print("Abrindo o gráfico final!")
plt.show()