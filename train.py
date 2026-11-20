"""Treina um classificador Naive Bayes para níveis de produtividade.

Uso:
    python train.py
    python train.py --csv caminho/para/social_media_vs_productivity.csv

Por padrão, o dataset é obtido pelo kagglehub. O script gera relatórios de EDA,
avalia as métricas no conjunto de teste e salva o pipeline completo com joblib.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "modelo_naive_bayes.pkl"

DATASET_SLUG = "mahdimashayekhi/social-media-vs-productivity"
TARGET_SCORE = "actual_productivity_score"
CLASS_ORDER = ["Baixa", "Média", "Alta"]

# A pontuação real é transformada em três faixas pelos tercis da própria base.
# Não usamos as duas pontuações de produtividade como preditoras para evitar
# vazamento da variável-alvo e uma previsão circular.
NUMERIC_FEATURES = [
    "age",
    "daily_social_media_time",
    "number_of_notifications",
    "work_hours_per_day",
    "stress_level",
    "sleep_hours",
    "screen_time_before_sleep",
    "breaks_during_work",
    "days_feeling_burnout_per_month",
    "weekly_offline_hours",
    "job_satisfaction_score",
]
CATEGORICAL_FEATURES = ["job_type"]
BOOLEAN_FEATURES = ["uses_focus_apps", "has_digital_wellbeing_enabled"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

BOOLEAN_MAP = {
    "true": True,
    "1": True,
    "yes": True,
    "y": True,
    "sim": True,
    "false": False,
    "0": False,
    "no": False,
    "n": False,
    "não": False,
    "nao": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento Naive Bayes - IFCE")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Caminho para um CSV já baixado. Se omitido, usa kagglehub.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fração destinada ao teste (padrão: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente de aleatoriedade para reprodutibilidade.",
    )
    return parser.parse_args()


def download_dataset() -> Path:
    """Baixa a base usando exatamente o identificador solicitado no enunciado."""
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "A dependência kagglehub não está instalada. Execute: "
            "pip install -r requirements.txt"
        ) from exc

    downloaded_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Caminho retornado pelo KaggleHub: {downloaded_path}")
    return Path(downloaded_path)


def locate_csv(path: Path) -> Path:
    """Localiza o CSV da base no diretório baixado."""
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"O arquivo informado não é CSV: {path}")
        return path

    if not path.exists():
        raise FileNotFoundError(f"Caminho do dataset não encontrado: {path}")

    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em: {path}")

    preferred = [
        file for file in csv_files
        if "social_media_vs_productivity" in file.name.lower()
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(csv_files) == 1:
        return csv_files[0]

    options = "\n".join(f" - {item}" for item in csv_files)
    raise RuntimeError(
        "Mais de um CSV foi encontrado e não foi possível identificar a base "
        f"correta automaticamente. Informe o arquivo com --csv.\n{options}"
    )


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normaliza colunas booleanas para True/False, preservando ausentes."""
    def convert(value: Any) -> Any:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return BOOLEAN_MAP.get(str(value).strip().lower(), np.nan)

    return series.map(convert)


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    initial_rows, initial_columns = df.shape
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    required_columns = set(ALL_FEATURES + [TARGET_SCORE])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "O CSV não contém todas as colunas esperadas. Colunas ausentes: "
            + ", ".join(missing_columns)
            + ". Confira se foi baixada a base 'Social Media vs Productivity'."
        )

    # Conversões explícitas: valores inválidos viram NaN e serão imputados.
    for column in NUMERIC_FEATURES + [TARGET_SCORE]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in BOOLEAN_FEATURES:
        df[column] = normalize_boolean(df[column])

    df["job_type"] = df["job_type"].astype("object")
    df["job_type"] = df["job_type"].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

    # Linhas sem o valor-alvo não podem participar de um treinamento supervisionado.
    before_target_drop = len(df)
    df = df.dropna(subset=[TARGET_SCORE]).copy()
    removed_missing_target = before_target_drop - len(df)

    report = {
        "initial_rows": int(initial_rows),
        "initial_columns": int(initial_columns),
        "duplicates_removed": duplicates,
        "rows_removed_missing_target": int(removed_missing_target),
        "rows_after_cleaning": int(len(df)),
        "columns": list(df.columns),
    }
    return df, report


def create_target(score: pd.Series) -> tuple[pd.Series, float, float]:
    """Cria classes Baixa/Média/Alta com os tercis da variável contínua."""
    q1, q2 = score.quantile([1 / 3, 2 / 3]).astype(float).tolist()
    if not q1 < q2:
        raise ValueError(
            "Não foi possível formar três classes distintas a partir de "
            f"{TARGET_SCORE}: os limites dos tercis coincidem ({q1}, {q2})."
        )

    labels = pd.cut(
        score,
        bins=[-np.inf, q1, q2, np.inf],
        labels=CLASS_ORDER,
        include_lowest=True,
        right=True,
    )
    return labels.astype(str), float(q1), float(q2)


def generate_eda(
    df: pd.DataFrame,
    y: pd.Series,
    data_report: dict[str, Any],
    q1: float,
    q2: float,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Estatísticas descritivas, ausências e distribuição das classes.
    df.describe(include="all").transpose().to_csv(
        REPORT_DIR / "estatisticas_descritivas.csv", encoding="utf-8-sig"
    )
    missing = df.isna().sum().rename("quantidade_ausente").to_frame()
    missing["percentual_ausente"] = (missing["quantidade_ausente"] / len(df) * 100).round(2)
    missing.to_csv(REPORT_DIR / "valores_ausentes.csv", encoding="utf-8-sig")

    class_counts = y.value_counts().reindex(CLASS_ORDER, fill_value=0)
    class_distribution = pd.DataFrame(
        {
            "classe": class_counts.index,
            "quantidade": class_counts.values,
            "percentual": (class_counts.values / len(y) * 100).round(2),
        }
    )
    class_distribution.to_csv(
        REPORT_DIR / "distribuicao_classes.csv", index=False, encoding="utf-8-sig"
    )

    numeric_for_corr = NUMERIC_FEATURES + [TARGET_SCORE]
    df[numeric_for_corr].corr(numeric_only=True).to_csv(
        REPORT_DIR / "correlacao_numerica.csv", encoding="utf-8-sig"
    )

    # Gráfico da distribuição da variável-alvo categórica.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(class_counts.index, class_counts.values)
    ax.set_title("Distribuição das classes de produtividade")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Quantidade de registros")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "distribuicao_classes.png", dpi=150)
    plt.close(fig)

    # Histogramas de alguns atributos numéricos para apoiar a exploração inicial.
    histogram_features = [
        "daily_social_media_time",
        "work_hours_per_day",
        "stress_level",
        "sleep_hours",
        "actual_productivity_score",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.flat, histogram_features):
        ax.hist(df[column].dropna(), bins=20, edgecolor="black", alpha=0.8)
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylabel("Frequência")
    axes.flat[-1].axis("off")
    fig.suptitle("Distribuição de variáveis selecionadas", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "histogramas_variaveis.png", dpi=150)
    plt.close(fig)

    lines = [
        "ANÁLISE EXPLORATÓRIA - SOCIAL MEDIA VS PRODUCTIVITY",
        "=" * 58,
        f"Arquivo CSV: {data_report.get('csv_path', 'não informado')}",
        f"Registros originais: {data_report['initial_rows']}",
        f"Colunas originais: {data_report['initial_columns']}",
        f"Duplicatas removidas: {data_report['duplicates_removed']}",
        f"Registros removidos por alvo ausente: {data_report['rows_removed_missing_target']}",
        f"Registros utilizados: {len(df)}",
        f"Quantidade de variáveis preditoras: {len(ALL_FEATURES)}",
        f"Variável-alvo original: {TARGET_SCORE}",
        f"Limite Baixa/Média (tercil 1): {q1:.4f}",
        f"Limite Média/Alta (tercil 2): {q2:.4f}",
        "",
        "VARIÁVEIS PREDITORAS:",
        *[f" - {feature}" for feature in ALL_FEATURES],
        "",
        "DISTRIBUIÇÃO DAS CLASSES:",
        *[
            f" - {row.classe}: {int(row.quantidade)} ({row.percentual:.2f}%)"
            for row in class_distribution.itertuples(index=False)
        ],
        "",
        "VALORES AUSENTES POR COLUNA:",
        *[
            f" - {column}: {int(count)}"
            for column, count in df.isna().sum().items()
        ],
        "",
        "Observação: os valores ausentes nas features são tratados pelo pipeline de",
        "pré-processamento (mediana para variáveis numéricas e moda para categóricas).",
    ]
    (REPORT_DIR / "relatorio_eda.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== ANÁLISE EXPLORATÓRIA ===")
    print(f"Dimensão original: {data_report['initial_rows']} linhas x {data_report['initial_columns']} colunas")
    print(f"Duplicatas removidas: {data_report['duplicates_removed']}")
    print(f"Registros utilizados: {len(df)}")
    print(f"Quantidade de variáveis preditoras: {len(ALL_FEATURES)}")
    print("\nEstatísticas descritivas:")
    print(df[NUMERIC_FEATURES + [TARGET_SCORE]].describe().round(3).to_string())
    print("\nValores ausentes por variável:")
    print(df[ALL_FEATURES + [TARGET_SCORE]].isna().sum().to_string())
    print("\nDistribuição das classes:")
    print(class_distribution.to_string(index=False))
    print(f"\nRelatórios de EDA salvos em: {REPORT_DIR}")


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES + BOOLEAN_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB()),
        ]
    )


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        raise SystemExit("--test-size deve estar entre 0.05 e 0.50.")

    source_path = args.csv if args.csv else download_dataset()
    csv_path = locate_csv(source_path).resolve()
    print(f"CSV selecionado: {csv_path}")

    df, data_report = load_and_clean_data(csv_path)
    data_report["csv_path"] = str(csv_path)

    y, q1, q2 = create_target(df[TARGET_SCORE])
    X = df[ALL_FEATURES].copy()
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)

    generate_eda(df, pd.Series(y), data_report, q1, q2)

    class_counts = pd.Series(y).value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            "Há uma classe com menos de dois exemplos; não é possível fazer "
            "uma divisão estratificada treino/teste."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_ORDER)

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix.tolist(),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
    }

    print("\n=== AVALIAÇÃO DO MODELO (CONJUNTO DE TESTE) ===")
    print(f"Acurácia:  {accuracy:.4f}")
    print(f"Precisão:  {precision:.4f} (macro)")
    print(f"Revocação: {recall:.4f} (macro)")
    print(f"F1-Score:  {f1:.4f} (macro)")
    print("\nMatriz de Confusão (linhas = classe real; colunas = prevista):")
    print(pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_string())
    print("\nRelatório por classe:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            zero_division=0,
        )
    )

    # O pipeline contém imputação, codificação e GaussianNB, permitindo que o
    # Flask aplique exatamente o mesmo pré-processamento usado no treinamento.
    job_type_options = sorted(
        str(value) for value in X["job_type"].dropna().unique()
    )
    artifact = {
        "pipeline": pipeline,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
        "target_source": TARGET_SCORE,
        "class_order": CLASS_ORDER,
        "target_thresholds": {"baixa_ate": q1, "media_ate": q2},
        "job_type_options": job_type_options,
        "metrics": metrics,
        "dataset_source": f"Kaggle: {DATASET_SLUG}",
        "dataset_csv": str(csv_path),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo e pré-processamento salvos em: {MODEL_PATH}")
    print("Treinamento concluído.")


if __name__ == "__main__":
    main()
