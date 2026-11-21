# Classificador de Produtividade — IFCE

Projeto final da disciplina **Probabilidade e Estatística**, do curso de **Engenharia da Computação do Instituto Federal do Ceará (IFCE)**. A aplicação utiliza o algoritmo **Gaussian Naive Bayes** para classificar o nível de produtividade em três categorias — **Baixa, Média e Alta** — a partir de hábitos digitais, características da rotina e informações de trabalho.

O projeto percorre as etapas de um fluxo de Machine Learning: obtenção dos dados, análise exploratória, preparação das variáveis, treinamento, avaliação, salvamento do modelo e disponibilização de previsões por uma aplicação web em Flask.

---

## 1. Visão geral do projeto

**Pergunta que o projeto investiga:** com base em informações sobre a rotina de uma pessoa, o modelo consegue estimar em qual faixa de produtividade ela se enquadra, segundo os padrões encontrados na base escolhida?

| Item | Definição |
|---|---|
| Base de dados | Social Media vs Productivity |
| Fonte | Kaggle — `mahdimashayekhi/social-media-vs-productivity` |
| Arquivo CSV | `social_media_vs_productivity.csv` |
| Algoritmo | Gaussian Naive Bayes (`GaussianNB`) |
| Tipo de problema | Classificação supervisionada multiclasse |
| Variável-alvo original | `actual_productivity_score` |
| Classes previstas | `Baixa`, `Média` e `Alta` |
| Quantidade de preditores | 14 |
| Divisão dos dados | 80% treinamento e 20% teste |
| Aplicação web | Flask + HTML + CSS + JavaScript |

> **Importante:** o modelo prevê uma faixa de produtividade conforme a distribuição da pontuação na base. Ele não mede diretamente a produtividade de uma pessoa, não fornece diagnóstico e não demonstra que redes sociais ou qualquer outro hábito cause determinado nível de produtividade.

## 2. Base de dados e análise exploratória

O treinamento baixa a base pelo KaggleHub usando o identificador solicitado no projeto:

```python
import kagglehub

path = kagglehub.dataset_download(
    "mahdimashayekhi/social-media-vs-productivity"
)
print("Path to dataset files:", path)
```

O script `train.py` procura o arquivo CSV dentro do diretório retornado. Também é possível informar um CSV já baixado manualmente, usando o argumento `--csv`.

### 2.1 Resultado da execução realizada

O relatório `reports/relatorio_eda.txt` gerado na execução com a base original registrou:

| Informação | Resultado |
|---|---:|
| Registros originais | 30.000 |
| Colunas originais | 19 |
| Duplicatas removidas | 0 |
| Registros removidos por ausência no alvo | 2.365 |
| Registros utilizados | 27.635 |
| Variáveis preditoras | 14 |

As 2.365 linhas removidas não possuíam valor em `actual_productivity_score`. Como essa coluna é a origem do rótulo que o modelo deve aprender a prever, não seria possível atribuir uma classe confiável a essas linhas.

### 2.2 Valores ausentes nas variáveis preditoras

O relatório da execução registrou valores ausentes nas seguintes features:

| Variável | Valores ausentes |
|---|---:|
| `daily_social_media_time` | 2.540 |
| `stress_level` | 1.755 |
| `sleep_hours` | 2.418 |
| `screen_time_before_sleep` | 2.042 |
| `job_satisfaction_score` | 2.499 |
| Demais nove features selecionadas | 0 |

Esses valores não exigem a exclusão de todas as linhas correspondentes. O pipeline de pré-processamento trata as ausências durante o treinamento: usa a **mediana** nas variáveis numéricas e a **moda** nas variáveis categóricas e booleanas. Os valores dos imputadores são aprendidos somente no conjunto de treinamento.

### 2.3 Arquivos produzidos pela análise exploratória

Depois de executar `python train.py`, a pasta `reports/` contém:

| Arquivo | Conteúdo |
|---|---|
| `relatorio_eda.txt` | Origem do CSV, dimensões, registros removidos, features, limites das classes, distribuição e ausências |
| `estatisticas_descritivas.csv` | Estatísticas descritivas das colunas da base |
| `valores_ausentes.csv` | Quantidade e percentual de valores ausentes por coluna |
| `distribuicao_classes.csv` | Quantidade e percentual de registros em cada classe |
| `distribuicao_classes.png` | Gráfico de distribuição das classes |
| `correlacao_numerica.csv` | Correlações entre as variáveis numéricas selecionadas e a pontuação original |
| `histogramas_variaveis.png` | Histogramas de variáveis numéricas selecionadas |

## 3. Definição da variável-alvo

A coluna `actual_productivity_score` é numérica, mas o trabalho pede um problema de **classificação supervisionada**. Para isso, o script converte a pontuação em três categorias, utilizando os tercis da distribuição:

- **Baixa:** pontuação menor ou igual ao primeiro tercil;
- **Média:** pontuação acima do primeiro tercil e menor ou igual ao segundo tercil;
- **Alta:** pontuação acima do segundo tercil.

Na execução registrada no relatório, os limites foram:

| Limite | Valor |
|---|---:|
| Primeiro tercil — Baixa/Média | 3,9146 |
| Segundo tercil — Média/Alta | 5,9998 |

Assim, nessa execução, as faixas ficaram definidas como:

- **Baixa:** `actual_productivity_score <= 3.9146`;
- **Média:** `3.9146 < actual_productivity_score <= 5.9998`;
- **Alta:** `actual_productivity_score > 5.9998`.

Os limites são recalculados quando o treinamento é executado. Se a versão da base ou a distribuição das pontuações mudar, os valores também poderão mudar. Os limites utilizados são salvos no artefato do modelo e exibidos na interface.

**Nota metodológica:** nesta implementação, os dois tercis são calculados sobre todos os valores válidos de `actual_productivity_score` antes da divisão entre treinamento e teste. Isso define uma referência única de classes para toda a base, mas faz com que a distribuição do alvo no conjunto de teste participe da definição dos limites. Em uma versão com protocolo ainda mais conservador, os limites poderiam ser calculados somente no conjunto de treinamento e depois aplicados ao conjunto de teste. Essa distinção deve ser conhecida ao interpretar as métricas.

### 3.1 Distribuição das classes

A execução registrada apresentou uma distribuição praticamente equilibrada:

| Classe | Registros | Percentual |
|---|---:|---:|
| Baixa | 9.212 | 33,33% |
| Média | 9.211 | 33,33% |
| Alta | 9.212 | 33,33% |
| **Total** | **27.635** | **100%** |

O equilíbrio é interessante porque reduz o risco de a avaliação ser dominada por uma classe muito mais frequente que as demais. Ainda assim, o equilíbrio das classes, por si só, não garante que o modelo tenha bom desempenho.

## 4. Variáveis preditoras (features)

Foram selecionadas 14 variáveis com informações sobre perfil, rotina profissional, hábitos digitais e bem-estar.

| Variável | Tipo | Justificativa da seleção |
|---|---|---|
| `age` | Numérica | Representa a faixa etária do participante. |
| `job_type` | Categórica | Contextualiza o tipo de ocupação. |
| `daily_social_media_time` | Numérica | Mede o tempo diário dedicado às redes sociais. |
| `number_of_notifications` | Numérica | Representa a quantidade de notificações e possíveis interrupções digitais. |
| `work_hours_per_day` | Numérica | Caracteriza a carga diária de trabalho. |
| `stress_level` | Numérica | Considera o nível de estresse informado. |
| `sleep_hours` | Numérica | Representa a duração do sono. |
| `screen_time_before_sleep` | Numérica | Mede o tempo de tela antes de dormir. |
| `breaks_during_work` | Numérica | Representa a quantidade de pausas durante o trabalho. |
| `uses_focus_apps` | Booleana | Indica se a pessoa utiliza aplicativos de foco. |
| `has_digital_wellbeing_enabled` | Booleana | Indica se o recurso de bem-estar digital está ativado. |
| `days_feeling_burnout_per_month` | Numérica | Representa a frequência mensal de sensação de esgotamento. |
| `weekly_offline_hours` | Numérica | Mede o tempo semanal sem conexão. |
| `job_satisfaction_score` | Numérica | Acrescenta uma medida de satisfação com o trabalho. |

### Por que algumas colunas não entram no modelo?

- `actual_productivity_score` é usada para construir a variável-alvo. Incluí-la entre as entradas seria fornecer ao modelo a própria pontuação da qual a classe é derivada.
- `perceived_productivity_score` foi excluída para evitar que outra avaliação direta de produtividade funcione como um atalho para a previsão.
- As demais colunas não foram selecionadas nesta versão porque o projeto prioriza um conjunto de 14 atributos com relação conceitual com rotina, hábitos digitais e trabalho.

A seleção é uma decisão metodológica do projeto; ela não significa que as variáveis escolhidas sejam causas comprovadas da produtividade.

## 5. Como o modelo funciona

O algoritmo utilizado é o **Gaussian Naive Bayes**, disponível no Scikit-Learn como `GaussianNB`.

O Naive Bayes aplica o Teorema de Bayes para estimar a probabilidade de cada classe a partir dos atributos informados. A versão *Gaussian* é apropriada para trabalhar com variáveis numéricas contínuas, pois modela a distribuição de cada atributo numérico por uma distribuição normal dentro de cada classe.

A principal hipótese do Naive Bayes é que as variáveis preditoras são condicionalmente independentes entre si quando a classe é conhecida. Na prática, características como sono, estresse e carga de trabalho podem ter relação entre si. Portanto, essa hipótese é uma simplificação do modelo e deve ser mencionada na apresentação.

### 5.1 Pré-processamento

O pipeline do Scikit-Learn reúne o pré-processamento e o classificador em um único objeto:

1. **Features numéricas:** valores ausentes preenchidos pela mediana.
2. **Features categóricas e booleanas:** valores ausentes preenchidos pela moda.
3. **Codificação:** `OneHotEncoder` transforma `job_type` e as duas variáveis booleanas em colunas numéricas. Categorias desconhecidas na previsão são ignoradas pelo codificador.
4. **Classificação:** o `GaussianNB` recebe os atributos processados e estima a classe.

Esse pipeline é importante porque a aplicação web utiliza exatamente o mesmo pré-processamento aprendido durante o treinamento. Assim, não é necessário implementar uma transformação diferente no Flask.

### 5.2 Por que Gaussian Naive Bayes?

As features do projeto incluem várias medidas numéricas, como horas de sono, tempo em redes sociais, notificações, estresse e horas de trabalho. O `GaussianNB` é uma escolha simples e didática para um problema multiclasse com esse perfil de dados.

Uma limitação que vale apresentar é que o modelo supõe independência condicional entre as features e aproxima variáveis transformadas em indicadores 0/1 por distribuições Gaussianas. O desempenho deve, portanto, ser avaliado empiricamente no conjunto de teste, em vez de presumido apenas pela escolha do algoritmo.

## 6. Treinamento e avaliação

O treinamento está implementado em `train.py`.

### 6.1 Divisão dos dados

O script separa os dados em:

- **80% para treinamento:** utilizados para ajustar o pipeline e o classificador;
- **20% para teste:** reservados para avaliar o desempenho em registros não utilizados no ajuste do modelo.

A divisão utiliza `stratify=y`, mantendo aproximadamente a proporção das classes nos dois conjuntos, e `random_state=42`, para tornar a separação reproduzível.

### 6.2 Métricas obrigatórias

Ao terminar o treinamento, o terminal apresenta as seguintes métricas:

| Métrica | O que informa |
|---|---|
| Acurácia | Proporção total de previsões corretas no conjunto de teste. |
| Precisão (macro) | Média da precisão calculada separadamente para cada classe, atribuindo o mesmo peso às três. |
| Revocação/Recall (macro) | Média da capacidade de identificar corretamente os registros de cada classe. |
| F1-Score (macro) | Média harmônica entre precisão e recall, calculada por classe e depois promediada. |
| Matriz de confusão | Mostra a quantidade de acertos e confusões entre classes. |

Na matriz de confusão impressa pelo script, **as linhas representam as classes reais** e **as colunas representam as classes previstas**. A ordem é `Baixa`, `Média` e `Alta`. Os valores da diagonal principal representam os acertos; os demais valores mostram em quais classes ocorreram as confusões.

As métricas são calculadas no conjunto de teste. O código também imprime um relatório detalhado por classe. Não registre valores de desempenho no relatório acadêmico antes de executar o treinamento e copiar os resultados reais exibidos no terminal.

## 7. Estrutura de diretórios

```text
projeto_n2_ifce/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── models/
│   └── modelo_naive_bayes.pkl   # criado após o treinamento
│
├── reports/                    # relatórios e gráficos da análise exploratória
│   ├── relatorio_eda.txt
│   ├── estatisticas_descritivas.csv
│   ├── valores_ausentes.csv
│   ├── distribuicao_classes.csv
│   ├── distribuicao_classes.png
│   ├── correlacao_numerica.csv
│   └── histogramas_variaveis.png
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── app.js
```

As pastas `models/` e `reports/` podem conter apenas arquivos `.gitkeep` antes da primeira execução. Os artefatos são gerados pelo treinamento.

## 8. Instalação e execução — passo a passo

Os comandos abaixo consideram Linux/Ubuntu, como no ambiente utilizado durante o desenvolvimento. Também há instruções para Windows.

### Passo 1 — Abrir a pasta do projeto

No terminal, entre na pasta que contém `train.py`, `app.py` e `requirements.txt`:

```bash
cd caminho/para/projeto_n2_ifce
```

No PyCharm, abra essa mesma pasta como projeto. Não execute os comandos a partir de dentro de `templates/` ou `static/`.

### Passo 2 — Criar e ativar o ambiente virtual

Se o ambiente `.venv` ainda não existir:

```bash
python3 -m venv .venv
```

Ative-o no Linux/Ubuntu:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o terminal geralmente mostra `(.venv)` antes do caminho.

### Passo 3 — Instalar as bibliotecas

Com o ambiente virtual ativado, execute:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Principais dependências:

- `pandas` e `numpy`: manipulação dos dados;
- `scikit-learn`: pré-processamento, Gaussian Naive Bayes e métricas;
- `kagglehub`: download da base;
- `joblib`: salvamento e carregamento do modelo;
- `matplotlib`: geração de gráficos da análise exploratória;
- `flask`: servidor web.

### Passo 4 — Baixar a base, analisar e treinar

Execute na raiz do projeto:

```bash
python train.py
```

Por padrão, o script:

1. baixa a base `mahdimashayekhi/social-media-vs-productivity` pelo KaggleHub;
2. localiza `social_media_vs_productivity.csv`;
3. verifica as colunas esperadas e remove duplicatas;
4. remove registros sem `actual_productivity_score`;
5. cria as três classes pelos tercis da pontuação;
6. gera os relatórios e gráficos exploratórios;
7. divide os dados em treino e teste;
8. ajusta o pré-processamento e o `GaussianNB`;
9. imprime as métricas e a matriz de confusão;
10. salva o pipeline treinado em `models/modelo_naive_bayes.pkl`.

**Alternativa — utilizar um CSV já baixado:**

```bash
python train.py --csv "/caminho/para/social_media_vs_productivity.csv"
```

Para alterar a proporção de teste, o script também aceita `--test-size`. Exemplo com 25% para teste:

```bash
python train.py --test-size 0.25
```

### Passo 5 — Conferir os resultados do treinamento

Antes de abrir o site, confira:

- o resumo da análise exploratória em `reports/relatorio_eda.txt`;
- a distribuição das classes em `reports/distribuicao_classes.csv` e `.png`;
- as estatísticas em `reports/estatisticas_descritivas.csv`;
- as métricas e a matriz de confusão impressas no terminal;
- a criação de `models/modelo_naive_bayes.pkl`.

O arquivo `.pkl` contém o pipeline treinado e metadados, incluindo a lista de features, os limites dos tercis, as categorias de ocupação, as métricas calculadas e a origem do dataset.

### Passo 6 — Iniciar a aplicação Flask

Somente depois de concluir o treinamento, execute:

```bash
python app.py
```

Abra o navegador em:

<http://127.0.0.1:5000>

Para encerrar o servidor, volte ao terminal e pressione `Ctrl + C`.

### Passo 7 — Fazer uma previsão

1. Preencha os 14 campos do formulário com informações de uma rotina.
2. Em **Tipo de ocupação**, informe ou selecione uma categoria disponível na lista sugerida.
3. Selecione `Sim` ou `Não` nos campos booleanos.
4. Clique em **Prever produtividade**.
5. A aplicação exibirá a classe prevista e as probabilidades estimadas para `Baixa`, `Média` e `Alta`.

A classe exibida é aquela com maior probabilidade estimada pelo classificador. As probabilidades não representam garantia de acerto.

## 9. Como a aplicação web funciona

O fluxo de previsão é:

```text
Usuário preenche o formulário
          ↓
HTML + JavaScript enviam os dados (POST /predict)
          ↓
Flask valida e organiza os 14 atributos
          ↓
Pipeline salvo aplica imputação e codificação
          ↓
Gaussian Naive Bayes calcula a previsão
          ↓
Flask retorna classe e probabilidades
          ↓
A página apresenta o resultado
```

- `app.py`: carrega o artefato com `joblib`, disponibiliza a rota `GET /` e recebe previsões em `POST /predict`.
- `templates/index.html`: contém o formulário e os painéis de resultado e explicação.
- `static/style.css`: define o layout e a responsividade.
- `static/app.js`: envia o formulário sem recarregar a página e apresenta a resposta.

O formulário possui campos correspondentes às mesmas 14 features usadas no treinamento. O Flask monta um DataFrame com esses nomes e envia os dados ao pipeline salvo.

## 10. Pontos interessantes para destacar na apresentação

1. **Transformação de regressão em classificação:** a base fornece uma pontuação numérica, mas o trabalho pede classificação. Os tercis permitem criar três faixas de produtividade.
2. **Classes equilibradas:** na execução registrada, cada classe representa aproximadamente um terço dos 27.635 registros utilizados.
3. **Tratamento de ausências sem descartar todos os registros:** as features incompletas são tratadas por imputação dentro do pipeline.
4. **Prevenção de vazamento direto do alvo:** a pontuação usada para construir as classes não entra como feature. A pontuação percebida também foi excluída.
5. **Mesmo pré-processamento no treino e na aplicação:** o pipeline inteiro é salvo com `joblib`, reduzindo diferenças entre o que o modelo aprende e o que recebe no Flask.
6. **Predição probabilística:** além da classe final, a interface mostra as probabilidades estimadas para as três classes.
7. **Avaliação multiclasse:** o uso de métricas macro permite observar o desempenho médio entre as classes, dando o mesmo peso a Baixa, Média e Alta.

### Limitações que a equipe deve conhecer

- **Associação não é causalidade:** o modelo identifica padrões estatísticos na base, mas não prova que reduzir redes sociais, dormir mais ou alterar outro hábito cause aumento de produtividade.
- **Hipótese de independência:** variáveis da rotina podem estar relacionadas entre si, enquanto o Naive Bayes assume independência condicional.
- **Hipótese Gaussiana:** `GaussianNB` modela as features transformadas por distribuições Gaussianas. Essa aproximação pode não representar perfeitamente todos os atributos, especialmente indicadores binários gerados pelo One-Hot Encoding.
- **Classes relativas à base:** os limites dos tercis descrevem a distribuição observada neste conjunto de dados; não são limites universais de produtividade.
- **Generalização:** as métricas dependem da base, da divisão treino/teste e das características dos dados. Uma boa pontuação no teste não garante o mesmo desempenho em outras populações.
- **Probabilidades estimadas:** a maior probabilidade indica a classe escolhida pelo modelo, mas não deve ser interpretada como certeza nem como probabilidade perfeitamente calibrada.

## 11. Solução de problemas comuns

### `TemplateNotFound: index.html`

O Flask procura o HTML dentro de `templates/`. Confirme esta estrutura:

```text
templates/index.html
static/style.css
static/app.js
```

Não deixe `index.html` diretamente na raiz do projeto.

### Mensagem informando que o modelo não foi encontrado

Execute primeiro:

```bash
python train.py
```

Confirme que o arquivo foi criado em `models/modelo_naive_bayes.pkl`. Depois inicie novamente `app.py`.

### `ModuleNotFoundError`

Confirme que o ambiente virtual está ativo e instale as dependências:

```bash
pip install -r requirements.txt
```

### Falha ao baixar a base pelo KaggleHub

Verifique a conexão com a internet, a instalação de `kagglehub` e as orientações de autenticação do KaggleHub/Kaggle aplicáveis à sua conta. Como alternativa, baixe o CSV e execute `train.py` com `--csv` apontando para o arquivo.

### A página abre, mas a previsão apresenta erro

Verifique se o treinamento terminou sem erro, se o modelo foi gerado pela versão atual do `train.py` e se todos os campos do formulário foram preenchidos. Após treinar um novo modelo, reinicie o servidor Flask para carregar o novo artefato.

## 12. Relação com os critérios do edital

| Critério do trabalho | Onde é atendido |
|---|---|
| Análise exploratória da base | `train.py` e arquivos em `reports/` |
| Seleção e justificativa das features | Seção 4 deste README |
| Treinamento e avaliação Naive Bayes | `train.py` e métricas no terminal |
| Salvamento do modelo | `models/modelo_naive_bayes.pkl` com `joblib` |
| Desenvolvimento do back-end | `app.py` com Flask |
| Desenvolvimento do front-end | `templates/index.html`, `static/style.css` e `static/app.js` |
| Documentação do projeto | Este README e os relatórios gerados |

## 13. Fonte da base de dados

Kaggle — [Social Media vs Productivity](https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity).

O projeto utiliza o identificador `mahdimashayekhi/social-media-vs-productivity` no KaggleHub. Os números de exploração apresentados neste documento correspondem ao relatório gerado na execução registrada com 30.000 registros originais; uma nova versão da base ou uma nova execução pode produzir limites e métricas diferentes.

---

**Projeto acadêmico — IFCE · Engenharia da Computação · Probabilidade e Estatística**