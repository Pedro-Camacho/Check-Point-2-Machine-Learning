# 🤖 Check Point 2 - Regressão Logística

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Projeto acadêmico de **Machine Learning** focado em **Regressão Logística** com técnicas de regularização (Lasso e Ridge) e experimentação com funções de perda alternativas.

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Objetivos](#-objetivos)
- [Dataset](#-dataset)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Resultados](#-resultados)
- [Equipe](#-equipe)
- [Licença](#-licença)
- [Contato](#-contato)

---

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido como parte da disciplina de **IA & Machine Learning** ministrada pelo Prof. Danilo Rodrigues de Assis Elias na **Escola Politécnica (3ESPW)**.

O trabalho explora técnicas fundamentais de Machine Learning aplicadas à classificação binária, com ênfase em:

- 📊 Análise exploratória de dados (EDA)
- 🔍 Validação cruzada estratificada
- 🎛️ Regularização L1 (Lasso) e L2 (Ridge)
- 🔧 Otimização de hiperparâmetros com Grid Search
- 🧪 Experimentação com funções de perda alternativas

### Por que este projeto é interessante?

- ✅ **Implementação completa**: Do pré-processamento à avaliação
- ✅ **Análise crítica**: Comparação fundamentada entre diferentes abordagens
- ✅ **Implementação manual**: Criação de algoritmo customizado com RMSE
- ✅ **Documentação rica**: Cada decisão é explicada e justificada
- ✅ **Didático**: Perfeito para aprendizado de ML

---

## 🎓 Objetivos

### Parte A: Análise Exploratória e Regularização

1. **Preparação e Análise Exploratória**
   - Carregar e explorar o dataset
   - Verificar balanceamento de classes
   - Analisar distribuição de features
   - Calcular matriz de correlação
   - Justificar necessidade de padronização

2. **Grid Search com Validação Cruzada**
   - Implementar pipeline de ML (StandardScaler + LogisticRegression)
   - Validação cruzada estratificada (k=5)
   - Testar regularização L1 e L2
   - Otimizar hiperparâmetro C

3. **Modelo Final**
   - Treinar com melhores parâmetros
   - Avaliar com múltiplas métricas
   - Gerar matriz de confusão

4. **Discussão**
   - Comparar coeficientes L1 vs L2
   - Identificar features eliminadas pelo Lasso
   - Discutir efeitos da regularização

### Parte B: Alterando a Função de Perda

1. **Experimento**
   - Implementar regressão logística com RMSE
   - Comparar com log-loss tradicional

2. **Comparação**
   - Avaliar acurácia e curva ROC
   - Analisar calibração de probabilidades

3. **Discussão Crítica**
   - Explicar por que RMSE é inadequado
   - Justificar superioridade da log-loss

---

## 📊 Dataset

**Nome:** Gender Recognition by Voice and Speech Analysis

**Fonte:** [Voice Gender Dataset](https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv)

### Características:

| Atributo | Valor |
|----------|-------|
| **Total de amostras** | 3,168 |
| **Features** | 20 características acústicas |
| **Classes** | 2 (masculino/feminino) |
| **Balanceamento** | Perfeito (50%/50%) |
| **Valores faltantes** | 0 |

### Features Principais:

- `meanfreq` - Frequência média (kHz)
- `sd` - Desvio padrão da frequência
- `median` - Mediana da frequência
- `Q25` - Primeiro quartil
- `Q75` - Terceiro quartil
- `IQR` - Intervalo interquartil
- `skew` - Assimetria
- `kurt` - Curtose
- `sfm` - Frequência média espectral
- `mode` - Moda da frequência
- E mais 10 features acústicas...

**Variável Alvo:** `label` (male/female)

---

## 📁 Estrutura do Projeto

```
Check-Point-2-Machine-Learning/
│
├── CP2_IA_and_ML.ipynb          # Notebook principal com todo o projeto
├── README.md                     # Este arquivo
├── requirements.txt              # Dependências do projeto (a criar)
└── .gitignore                    # Arquivos ignorados pelo Git
```

### Estrutura do Notebook:

```
CP2_IA_and_ML.ipynb
│
├── PARTE A: Análise Exploratória, Validação Cruzada, Lasso e Ridge
│   │
│   ├── 1. Preparação e Análise Exploratória
│   │   ├── Carregamento dos dados
│   │   ├── Análise de balanceamento
│   │   ├── Distribuição de features (histogramas)
│   │   ├── Matriz de correlação
│   │   └── Discussão sobre multicolinearidade
│   │
│   ├── 2. Grid Search com Validação Cruzada
│   │   ├── Configuração do pipeline
│   │   ├── StandardScaler + LogisticRegression
│   │   ├── StratifiedKFold (k=5)
│   │   ├── Teste de penalizações (L1, L2)
│   │   ├── Teste de valores de C
│   │   └── Seleção do melhor modelo
│   │
│   ├── 3. Modelo Final
│   │   ├── Retreinamento com melhores parâmetros
│   │   ├── Predições no conjunto de teste
│   │   ├── Cálculo de métricas
│   │   └── Matriz de confusão
│   │
│   └── 4. Discussão
│       ├── Comparação de coeficientes L1 vs L2
│       ├── Features eliminadas pelo Lasso
│       └── Efeitos da regularização
│
└── PARTE B: Alterando a Função de Perda
    │
    ├── 1. Experimento
    │   ├── Implementação de LogisticRegressionRMSE
    │   ├── Função sigmoid com proteção numérica
    │   ├── Gradiente descendente manual
    │   └── Treinamento do modelo
    │
    ├── 2. Comparação
    │   ├── Acurácia: Log-Loss vs RMSE
    │   ├── AUC-ROC
    │   ├── Curva ROC
    │   ├── Brier Score
    │   └── Curva de calibração
    │
    └── 3. Discussão Crítica
        ├── Por que RMSE é inadequado?
        ├── Problemas de calibração
        ├── Falta de fundamento probabilístico
        └── Vantagens da log-loss
```

---

## 🛠️ Tecnologias Utilizadas

### Linguagem
- **Python 3.8+**

### Bibliotecas Principais

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `numpy` | 1.21+ | Computação numérica |
| `pandas` | 1.3+ | Manipulação de dados |
| `matplotlib` | 3.4+ | Visualização de dados |
| `seaborn` | 0.11+ | Visualização estatística |
| `scikit-learn` | 1.0+ | Modelos de ML e métricas |

### Ferramentas
- **Jupyter Notebook** - Ambiente de desenvolvimento interativo
- **Git** - Controle de versão

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook

### Passo a Passo

1. **Clone o repositório**

```bash
git clone https://github.com/Pedro-Camacho/Check-Point-2-Machine-Learning.git
cd Check-Point-2-Machine-Learning
```

2. **Crie um ambiente virtual (recomendado)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instale as dependências**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Ou crie um arquivo `requirements.txt` com:

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

E instale com:

```bash
pip install -r requirements.txt
```

4. **Inicie o Jupyter Notebook**

```bash
jupyter notebook
```

5. **Abra o arquivo `CP2_IA_and_ML.ipynb`**

---

## 💻 Como Usar

### Execução Completa

1. Abra o notebook `CP2_IA_and_ML.ipynb`
2. Execute todas as células sequencialmente: `Cell > Run All`
3. Aguarde o processamento completo (~2-5 minutos)
4. Explore os resultados e visualizações

### Execução por Partes

#### Parte A - Regressão Logística Tradicional

```python
# Execute as células da seção "Parte A"
# Isso inclui:
# - Análise exploratória
# - Grid Search
# - Modelo final
# - Comparação L1 vs L2
```

#### Parte B - Experimento com RMSE

```python
# Execute as células da seção "Parte B"
# Isso inclui:
# - Implementação customizada
# - Comparação de modelos
# - Análise crítica
```

### Modificando o Projeto

#### Testar com diferentes hiperparâmetros:

```python
# Na célula de Grid Search, modifique:
param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Mais valores
    'model__solver': ['liblinear']
}
```

#### Alterar a validação cruzada:

```python
# Modifique o k-fold:
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # k=10
```

#### Usar outras métricas:

```python
# No Grid Search, altere o scoring:
grid = GridSearchCV(
    pipe, 
    param_grid, 
    cv=cv, 
    scoring='f1',  # ou 'precision', 'recall', 'roc_auc'
    n_jobs=-1
)
```

---

## 📈 Resultados

### Principais Descobertas

#### 1. Balanceamento Perfeito
- Dataset idealmente balanceado (50%/50%)
- Não requer técnicas de balanceamento (SMOTE, etc.)

#### 2. Feature Mais Importante
- **`meanfun`** (frequência média fundamental) é a feature dominante
- Coeficientes: ~-5.08 (L1) e ~-4.82 (L2)

#### 3. Desempenho L1 vs L2

| Métrica | L1 (Lasso) | L2 (Ridge) |
|---------|-----------|-----------|
| Acurácia | ~97%+ | ~97%+ |
| Features Zeradas | Algumas | Nenhuma |
| Interpretabilidade | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

#### 4. Log-Loss vs RMSE

| Aspecto | Log-Loss | RMSE |
|---------|----------|------|
| Calibração | ✅ Excelente | ❌ Ruim |
| Convergência | ✅ Garantida | ⚠️ Instável |
| Base Teórica | ✅ Sólida | ❌ Inexistente |
| Performance | ✅ Superior | ⚠️ Inferior |

### Visualizações Geradas

- ✅ Histogramas de distribuição de features
- ✅ Matriz de correlação (heatmap)
- ✅ Matriz de confusão
- ✅ Curva ROC
- ✅ Curva de calibração
- ✅ Comparação de coeficientes L1 vs L2

### Conclusões Principais

1. **Regularização L1 (Lasso)**
   - Ideal quando há features irrelevantes
   - Produz modelos esparsos (coeficientes zerados)
   - Melhor para interpretabilidade

2. **Regularização L2 (Ridge)**
   - Ideal quando há multicolinearidade
   - Mantém todas as features com pesos reduzidos
   - Mais estável numericamente

3. **RMSE para Classificação**
   - ❌ Não é adequado para problemas de classificação
   - ❌ Probabilidades mal calibradas
   - ❌ Sem fundamento em máxima verossimilhança
   - ✅ Log-loss é teoricamente superior

---

## 👥 Equipe

Este projeto foi desenvolvido por estudantes da **Escola Politécnica (3ESPW)**:

| Nome | RM | GitHub |
|------|-----|--------|
| **Camila Pedroza da Cunha** | 558768 | - |
| **Isabelle Dallabeneta Carlesso** | 554592 | - |
| **Nicoli Amy Kassa** | 559104 | - |
| **Pedro Almeida e Camacho** | 556831 | [@Pedro-Camacho](https://github.com/Pedro-Camacho) |
| **Renan Dias Utida** | 558540 | - |

**Professor:** Danilo Rodrigues de Assis Elias  
**Disciplina:** IA & Machine Learning

---

## 📚 Referências

### Artigos e Papers
- Tibshirani, R. (1996). "Regularization and variable selection via the lasso"
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"

### Livros
- **An Introduction to Statistical Learning** - James, Witten, Hastie, Tibshirani
- **Pattern Recognition and Machine Learning** - Christopher Bishop
- **Hands-On Machine Learning** - Aurélien Géron

### Documentação
- [Scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Dataset
- [Voice Gender Dataset](https://www.kaggle.com/primaryobjects/voicegender) - Kaggle
- [Original Repository](https://github.com/primaryobjects/voice-gender)

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

### Sugestões de Melhorias

- [ ] Adicionar mais visualizações (curvas de aprendizagem)
- [ ] Implementar outras técnicas de regularização (Elastic Net)
- [ ] Testar com outros datasets
- [ ] Criar interface interativa (Streamlit/Gradio)
- [ ] Adicionar testes unitários
- [ ] Documentar funções com docstrings
- [ ] Criar módulos Python reutilizáveis

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

```
MIT License

Copyright (c) 2024 Pedro Almeida e Camacho

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contato

**Pedro Almeida e Camacho**
- GitHub: [@Pedro-Camacho](https://github.com/Pedro-Camacho)
- RM: 556831

**Link do Projeto:** [https://github.com/Pedro-Camacho/Check-Point-2-Machine-Learning](https://github.com/Pedro-Camacho/Check-Point-2-Machine-Learning)

---

## 🌟 Agradecimentos

- Prof. Danilo Rodrigues de Assis Elias pela orientação
- Escola Politécnica pelo suporte acadêmico
- Comunidade scikit-learn pela excelente biblioteca
- Autores do dataset Voice Gender

---

## 📊 Estatísticas do Projeto

![GitHub repo size](https://img.shields.io/github/repo-size/Pedro-Camacho/Check-Point-2-Machine-Learning)
![GitHub contributors](https://img.shields.io/github/contributors/Pedro-Camacho/Check-Point-2-Machine-Learning)
![GitHub stars](https://img.shields.io/github/stars/Pedro-Camacho/Check-Point-2-Machine-Learning?style=social)
![GitHub forks](https://img.shields.io/github/forks/Pedro-Camacho/Check-Point-2-Machine-Learning?style=social)

---

<div align="center">

**⭐ Se este projeto foi útil para você, considere dar uma estrela!**

Feito com ❤️ pela equipe 3ESPW

[⬆ Voltar ao topo](#-check-point-2---regressão-logística)

</div>
