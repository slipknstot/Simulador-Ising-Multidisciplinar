import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. A Física (Metropolis otimizado para animação)
# ==========================================
def metropolis_vectorized(grid, beta):
    """
    Executa uma varredura completa na grade utilizando o algoritmo de Metropolis vetorizado (Checkerboard).

    Args:
        grid (numpy.ndarray): Matriz 2D representando os spins (+1 ou -1).
        beta (float): Parâmetro físico correspondente ao inverso da temperatura (1/T).

    Returns:
        numpy.ndarray: Grade atualizada após uma iteração completa de Monte Carlo.
    """
    N = grid.shape[0]
    x, y = np.indices((N, N))
    checker_white = (x + y) % 2 == 0
    checker_black = (x + y) % 2 != 0

    for mask in [checker_white, checker_black]:
        vizinhos = (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                    np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))
        dE = 2 * grid * vizinhos
        prob = np.exp(-dE * beta)
        rand_matrix = np.random.rand(N, N)
        aceitar = (dE <= 0) | (rand_matrix < prob)
        grid[mask & aceitar] *= -1
    return grid

# ==========================================
# 2. Configuração do Cenário
# ==========================================
N = 50 # Matriz 50x50 (2500 átomos em cada tela)

# Começamos com duas grades totalmente aleatórias
grid_frio = np.random.choice([-1, 1], size=(N, N))
grid_quente = np.random.choice([-1, 1], size=(N, N))

# Temperaturas
beta_frio = 1.0 / 1.0  # T = 1.0 (Muito abaixo do Ponto Crítico)
beta_quente = 1.0 / 4.0 # T = 4.0 (Muito acima do Ponto Crítico)

# Criando a janela com 2 gráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Simulação do Modelo de Ising em Tempo Real", fontsize=14)

# Renderizando as imagens iniciais
img_frio = ax1.imshow(grid_frio, cmap='coolwarm', vmin=-1, vmax=1)
img_quente = ax2.imshow(grid_quente, cmap='coolwarm', vmin=-1, vmax=1)

ax1.set_title("Temperatura Fria ($T=1.0$) \n Formação de Ordem")
ax2.set_title("Temperatura Quente ($T=4.0$) \n Caos Térmico")
ax1.axis('off') # Tira os números dos eixos para ficar mais bonito
ax2.axis('off')

# ==========================================
# 3. O Motor da Animação
# ==========================================
def atualizar_frame(frame):
    global grid_frio, grid_quente
    # Atualizado para o novo nome padronizado
    grid_frio = metropolis_vectorized(grid_frio, beta_frio)
    grid_quente = metropolis_vectorized(grid_quente, beta_quente)
    
    img_frio.set_data(grid_frio)
    img_quente.set_data(grid_quente)
    return img_frio, img_quente

# ==========================================
# 4. Rodando o "Filme"
# ==========================================
print("Iniciando a animação... Feche a janela do gráfico para encerrar.")

# FuncAnimation é quem chama a função 'atualizar_frame' repetidamente
ani = animation.FuncAnimation(
    fig, 
    atualizar_frame, 
    frames=10000,      # Número de quadros 
    interval=50,     # Tempo entre cada quadro em milissegundos (50ms = 20 fps)
    blit=True        # Otimiza o desenho na tela
)

plt.tight_layout()
plt.show()
