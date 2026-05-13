import numpy as np
import matplotlib.pyplot as plt

# 1. Nossa função super-rápida (já testada!)
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

# 2. Funções Matemáticas (Novidade!)
def calcular_energia(grid):
    vizinhos = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
    # Dividimos por 2 para não contar as interações dos vizinhos duas vezes
    return -0.5 * np.sum(grid * vizinhos)

def calcular_magnetizacao(grid):
    # É literalmente a soma de todos os spins (+1 e -1)
    return np.sum(grid)

# 3. O Experimento
N = 20  # Tamanho da matriz
temperaturas = np.linspace(1.5, 3.5, 30) # Focando no intervalo crítico
passos_equilibrio = 500  # Passos para o sistema estabilizar
passos_medicao = 1000    # Passos onde vamos fazer as anotações

# Listas para guardar os resultados finais de cada temperatura
C_v_lista = []
chi_lista = []

print("Calculando grandezas termodinâmicas... Aguarde.")

for T in temperaturas:
    beta = 1.0 / T
    grid = np.random.choice([-1, 1], size=(N, N))
    
    # Deixa o sistema rodar um pouco para estabilizar
    for _ in range(passos_equilibrio):
        grid = metropolis_vectorized(grid, beta)
        
    E_anotacoes = []
    M_anotacoes = []
    
    # Agora sim, medimos a energia e magnetização!
    for _ in range(passos_medicao):
        grid = metropolis_vectorized(grid, beta)
        E_anotacoes.append(calcular_energia(grid))
        M_anotacoes.append(abs(calcular_magnetizacao(grid)))
        
    # Aplica as fórmulas de variância (Estatística Pura)
    C_v = np.var(E_anotacoes) / (T**2)
    chi = np.var(M_anotacoes) / T
    
    C_v_lista.append(C_v)
    chi_lista.append(chi)

print("Cálculos concluídos! Desenhando gráficos.")

# 4. Plotando a Prova Científica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico da Capacidade Térmica
ax1.plot(temperaturas, C_v_lista, 'o-', color='darkorange')
ax1.axvline(x=2.269, color='k', linestyle='--', label='T_c Teórico (2.269)')
ax1.set_title("Capacidade Térmica ($C_v$)")
ax1.set_xlabel("Temperatura")
ax1.legend()

# Gráfico da Suscetibilidade Magnética
ax2.plot(temperaturas, chi_lista, 'o-', color='purple')
ax2.axvline(x=2.269, color='k', linestyle='--', label='T_c Teórico (2.269)')
ax2.set_title("Suscetibilidade Magnética ($\chi$)")
ax2.set_xlabel("Temperatura")
ax2.legend()

plt.tight_layout()
plt.show()