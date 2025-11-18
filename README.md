📘 Projeto Individual 1 (PI1) — Mineração de Dados (Supervisionado)

Este projeto foi feito para aplicar os conceitos aprendidos na disciplina de Mineração de Dados, do curso de Sistemas de Informação, ele aplica técnicas de aprendizado de máquina supervisionado para resolver dois problemas distintos utilizando Python e dados fictícios. O foco é demonstrar o fluxo completo: ETL → análise exploratória → modelagem → avaliação → interpretação.

📌 1. Descrição do Problema

- O projeto utiliza um dataset fictício de uma empresa de serviços digitais que deseja:
- Classificar se um cliente tem risco de churn utilizando Random Forest.
- Estimar o tempo médio de uso semanal (horas) por meio de Regressão Linear.
- Ambos os cenários refletem problemas comuns em análise de clientes e retenção.

🧹 2. ETL — Tratamento de Dados

- Os dados foram gerados artificialmente de forma realista.
- Remoção de duplicatas para evitar viés.
- Variáveis numéricas passaram por preparação adequada.
- Para a regressão foi aplicada normalização com StandardScaler.
- O objetivo foi garantir que os dados estivessem prontos para os modelos supervisionados.

📊 3. Exploração e Visualização

- A etapa exploratória mostrou:
- Satisfação e número de tickets abertos influenciam diretamente o churn.
- O heatmap revelou correlações moderadas entre variáveis de comportamento.
- A variável alvo churn apresentou leve desbalanceamento, mas ainda aceitável para Random Forest.
- As visualizações ajudaram a entender a estrutura dos dados antes da modelagem.

🌲 4. Resultados — Random Forest (Classificação de Churn)

- A acurácia ficou entre 75% e 90%, compatível com problemas de churn reais.
- As variáveis mais importantes foram:
- satisfação
- tickets_abertos
- tempo_de_plano

Esses fatores fazem sentido: clientes insatisfeitos e que enfrentam muitos problemas tendem a cancelar.

📈 5. Resultados — Regressão Linear (Uso Semanal)

- O MSE apresentou erro moderado, adequado ao cenário com ruído.
- O R² ficou entre 0.45 e 0.65, indicando boa explicabilidade parcial.
- O comportamento do cliente não é totalmente linear, mas o modelo fornece estimativas úteis.

🛠 Tecnologias Utilizadas:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
