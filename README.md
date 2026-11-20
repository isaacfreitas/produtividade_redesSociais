# Classificador de Produtividade — IFCE

Projeto de Machine Learning de ponta a ponta para o Trabalho Final de Probabilidade e Estatística (Engenharia da Computação/IFCE), usando a base **Social Media vs Productivity** e o algoritmo **Gaussian Naive Bayes**.

## 1. Problema de classificação

A base contém uma pontuação numérica chamada `actual_productivity_score`. Como o edital exige classificação supervisionada, o projeto transforma essa pontuação em três classes:

- **Baixa:** pontuação até o primeiro tercil da base;
- **Média:** pontuação acima do primeiro tercil e até o segundo tercil;
- **Alta:** pontuação acima do segundo tercil.

Os dois limites são calculados a partir dos dados carregados e salvos junto ao modelo em `models/modelo_naive_bayes.pkl`. Assim, as classes ficam definidas de maneira reproduzível para a versão da base usada no treinamento. Os limites numéricos e a distribuição efetiva são impressos no terminal e registrados em `reports/`.

### Features selecionadas

| Variável | Justificativa |
|---|---|
| `age` | Representa o perfil etário do participante. |
| `job_type` | Contextualiza a rotina profissional ou acadêmica. |
| `daily_social_media_time` | Mede o tempo diário dedicado às redes sociais. |
| `number_of_notifications` | Representa a intensidade de interrupções digitais. |
| `work_hours_per_day` | Caracteriza a carga diária de trabalho. |
| `stress_level` | Representa o nível de estresse informado. |
| `sleep_hours` | Considera a duração do sono. |
| `screen_time_before_sleep` | Mede o uso de telas antes de dormir. |
| `breaks_during_work` | Representa a frequência de pausas durante o trabalho. |
| `uses_focus_apps` | Indica o uso de aplicativos de foco. |
| `has_digital_wellbeing_enabled` | Indica o uso de recurso de bem-estar digital. |
| `days_feeling_burnout_per_month` | Representa a frequência mensal de sensação de esgotamento. |
| `weekly_offline_hours` | Mede o tempo semanal desconectado. |
| `job_satisfaction_score` | Acrescenta uma medida de satisfação com o trabalho. |

`actual_productivity_score` é a origem do alvo e **não** é usada como preditora. `perceived_productivity_score` também foi excluída para evitar que uma avaliação subjetiva de produtividade funcione como atalho para prever outra medida de produtividade.

## 2. Estrutura de diretórios

```text
projeto_n2_ifce/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── models/
│   └── modelo_naive_bayes.pkl     # gerado pelo train.py
├── reports/                      # relatórios e gráficos gerados pelo treino
├── static/
│   ├── app.js
│   └── style.css
└── templates/
    └── index.html
```

## 3. Instalação

Recomendado: Python 3.10 ou superior.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Obtenção e análise da base

Por padrão, `train.py` executa o download com KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("mahdimashayekhi/social-media-vs-productivity")
print("Path to dataset files:", path)
```

O script procura o CSV dentro do diretório baixado, verifica as colunas necessárias, remove registros duplicados, converte os tipos e remove linhas sem a variável-alvo. Valores ausentes nas features não precisam ser apagados: eles são imputados dentro do pipeline para evitar vazamento entre treino e teste.

Para treinar:

```bash
python train.py
```

Caso já tenha baixado o arquivo CSV manualmente:

```bash
python train.py --csv caminho/para/social_media_vs_productivity.csv
```

O script gera em `reports/`:

- `relatorio_eda.txt`: dimensão, duplicatas, features, limites das classes e valores ausentes;
- `estatisticas_descritivas.csv`: estatísticas descritivas;
- `valores_ausentes.csv`: quantidade e percentual de ausências;
- `distribuicao_classes.csv` e `distribuicao_classes.png`: frequência das três classes;
- `correlacao_numerica.csv`: correlação entre as variáveis numéricas selecionadas e a pontuação original;
- `histogramas_variaveis.png`: histogramas de variáveis relevantes.

## 5. Treinamento e avaliação

O `train.py` realiza uma divisão estratificada de 80% para treinamento e 20% para teste (semente aleatória 42). O pré-processamento fica dentro de um `Pipeline` do Scikit-Learn:

- Variáveis numéricas: imputação pela mediana;
- Variáveis categóricas/booleanas: imputação pela moda e One-Hot Encoding;
- Classificador: `GaussianNB`.

Ao final, o terminal apresenta:

- Acurácia;
- Precisão macro;
- Revocação macro;
- F1-Score macro;
- Matriz de confusão;
- Relatório de desempenho por classe.

A média macro dá o mesmo peso às três classes. O modelo completo, incluindo pré-processamento e metadados, é salvo em `models/modelo_naive_bayes.pkl` com `joblib`.

## 6. Executar a aplicação web

Depois de treinar o modelo:

```bash
python app.py
```

Abra no navegador: <http://127.0.0.1:5000>

O Flask carrega o artefato salvo, recebe os valores do formulário, monta um DataFrame com as mesmas features do treinamento, executa o pipeline e retorna a classe prevista e as probabilidades estimadas para cada classe.

## 7. Observações metodológicas importantes

1. Os tercis são calculados usando os valores de `actual_productivity_score` da base disponível no treinamento. Portanto, os limites podem mudar caso a versão do CSV mude.
2. O conjunto de teste é separado antes do ajuste do pipeline de pré-processamento; imputadores e codificador são ajustados apenas no conjunto de treinamento.
3. As probabilidades do Naive Bayes são estimativas do modelo, não garantias de acerto.
4. A base deve ser interpretada de acordo com sua origem e metodologia. Padrões encontrados nela não demonstram causalidade entre redes sociais e produtividade.
5. O arquivo `.pkl` deve ser carregado apenas quando tiver sido gerado por você ou por uma fonte confiável, pois arquivos pickle/joblib podem executar código durante a desserialização.

## 8. Relação com os critérios do edital

| Exigência | Implementação |
|---|---|
| Análise exploratória | `train.py` + arquivos em `reports/` |
| Seleção e justificativa das features | Seção 1 deste README |
| Naive Bayes e métricas | `train.py` com GaussianNB e as métricas exigidas |
| Salvamento do modelo | `models/modelo_naive_bayes.pkl` |
| Back-end | `app.py` com Flask |
| Front-end | `templates/index.html`, `static/style.css` e `static/app.js` |
| Documentação | Este README |

## Fonte da base

Kaggle — [Social Media vs Productivity](https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity).
