# Simulador-Ising-Multidisciplinar
Um laboratório computacional construído em Python para explorar transições de fase da matéria, unindo modelagem numérica de alta performance e algoritmos de Aprendizado Não Supervisionado.

Este repositório documenta minha jornada de exploração interdisciplinar. Como busco entender sistemas complexos antes de iniciar a graduação em Meteorologia na universidade federal, utilizo este projeto para investigar ativamente as fronteiras entre a física quântica, as energias renováveis e a ciência de dados — afinal, a matemática por trás da organização dos átomos é a mesma que rege o caos atmosférico e a eficiência de novos materiais.

## Tecnologias Utilizadas
* **Python 3**
* **NumPy:** Vetorização e computação de alta performance (Checkerboard Algorithm).
* **Matplotlib:** Renderização de animações em tempo real e visualização de dados gráficos.
* **Scikit-Learn:** Aplicação de Análise de Componentes Principais (PCA).

## Arquitetura do Projeto

O projeto foi dividido em três abordagens fundamentais para analisar o mesmo sistema complexo:

### 1. O Visual: Simulação Dinâmica (`Visual_Animation.py`)
Implementação do Algoritmo de Metropolis-Hastings. O código gera uma interface lado a lado simulando o comportamento de spins atômicos em temperaturas abaixo e acima do ponto crítico ($T_c$). Utilizamos vetorização via NumPy (padrão tabuleiro de xadrez) para otimizar o tempo de execução, permitindo rodar simulações em grande escala em tempo real.

### 2. O Analítico: Métricas Termodinâmicas (`Termodinamica.py`)
Prova matemática da transição de fase utilizando a mecânica estatística clássica. O script coleta amostras de energia e magnetização para calcular e plotar a Capacidade Térmica ($C_v$) e a Suscetibilidade Magnética ($\chi$), evidenciando o comportamento assintótico exatamente na temperatura crítica teórica ($T \approx 2.269$).

### 3. O Algorítmico: Machine Learning (`3_Machine_Learning_PCA.py`)
A cereja do bolo. Aplicamos um algoritmo de aprendizado não supervisionado (PCA) sobre milhares de matrizes brutas do sistema. Sem receber qualquer informação prévia sobre as regras da física ou sobre o conceito de temperatura, a Inteligência Artificial é capaz de extrair a "ordem" do sistema e redescobrir, de forma autônoma, o ponto exato da quebra de simetria da matéria.

## Como Executar na Sua Máquina

1. Clone este repositório:
 ```bash
   git clone [https://github.com/slipknstot/Simulador-Ising-Multidisciplinar.git](https://github.com/slipknstot/Simulador-Ising-Multidisciplinar.git)
