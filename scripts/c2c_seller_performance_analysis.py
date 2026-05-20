from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor


def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower=lower, upper=upper)


def save_distribution_plots(data, raw_cols, log_cols, output_path):
    fig, axes = plt.subplots(nrows=2, ncols=len(raw_cols), figsize=(18, 8))

    for idx, col in enumerate(raw_cols):
        sns.histplot(data[col], bins=30, kde=True, ax=axes[0, idx], color="#4C78A8")
        axes[0, idx].set_title(f"{col} (raw)")
        axes[0, idx].set_xlabel("")

    for idx, col in enumerate(log_cols):
        sns.histplot(data[col], bins=30, kde=True, ax=axes[1, idx], color="#F58518")
        axes[1, idx].set_title(f"{col} (log1p)")
        axes[1, idx].set_xlabel("")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_boxplots(data, raw_cols, log_cols, output_path):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    sns.boxplot(data=data[raw_cols], orient="h", ax=axes[0], palette="Set2")
    axes[0].set_title("Raw Variable Boxplots", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Raw values")

    sns.boxplot(data=data[log_cols], orient="h", ax=axes[1], palette="Set2")
    axes[1].set_title("Log-Transformed Variable Boxplots", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Log values: ln(x + 1)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_outputs(data, corr_cols, matrix_csv_path, heatmap_png_path):
    corr_matrix = data[corr_cols].corr(method="pearson")
    corr_matrix.to_csv(matrix_csv_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(heatmap_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_descriptive_statistics_table(data, variables):
    desc = data[variables].agg(["count", "mean", "std", "min", "max"]).T.reset_index()
    desc.columns = ["variable", "n", "mean", "std_dev", "min", "max"]
    desc["n"] = desc["n"].astype(int)
    return desc


def descriptive_stats_to_text(table):
    rounded = table.copy()
    for col in ["mean", "std_dev", "min", "max"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.4f}")
    rounded["n"] = rounded["n"].astype(str)
    return rounded.to_string(index=False)


def build_vif_table(model_data, vif_features):
    vif_input = model_data[vif_features].copy()
    vif_input.insert(0, "intercept", 1.0)

    vif_table = pd.DataFrame(
        {
            "variable": vif_input.columns,
            "vif": [
                variance_inflation_factor(vif_input.values, idx)
                for idx in range(vif_input.shape[1])
            ],
        }
    )
    return vif_table


def save_model_outputs(model, summary_txt_path, coef_csv_path):
    summary_txt_path.write_text(model.summary().as_text(), encoding="utf-8")

    coef_table = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t_or_z": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": model.conf_int()[0],
            "ci_upper": model.conf_int()[1],
        }
    )
    coef_table.to_csv(coef_csv_path, encoding="utf-8-sig")


def save_diagnostic_text(vif_table, bp_test, white_test, output_path):
    lines = [
        "Multicollinearity Test (VIF)",
        vif_table.to_string(index=False),
        "",
        "Breusch-Pagan Test",
        f"LM Statistic: {bp_test[0]:.6f}",
        f"LM p-value: {bp_test[1]:.6g}",
        f"F Statistic: {bp_test[2]:.6f}",
        f"F p-value: {bp_test[3]:.6g}",
        "",
        "White Test",
        f"LM Statistic: {white_test[0]:.6f}",
        f"LM p-value: {white_test[1]:.6g}",
        f"F Statistic: {white_test[2]:.6f}",
        f"F p-value: {white_test[3]:.6g}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_country_subsample_models(model_data, formula, output_csv_path, top_n=3):
    results = []
    top_countries = model_data["country"].value_counts().head(top_n).index.tolist()

    for country in top_countries:
        subset = model_data.loc[model_data["country"] == country].copy()
        if len(subset) < 500:
            continue

        model = smf.ols(formula=formula, data=subset).fit(cov_type="HC3")
        results.append(
            {
                "country": country,
                "n_obs": len(subset),
                "r_squared": model.rsquared,
                "coef_products_listed_log": model.params.get("productsListed_log", np.nan),
                "p_products_listed_log": model.pvalues.get("productsListed_log", np.nan),
                "coef_followers_log": model.params.get("socialNbFollowers_log", np.nan),
                "p_followers_log": model.pvalues.get("socialNbFollowers_log", np.nan),
                "coef_likes_log": model.params.get("socialProductsLiked_log", np.nan),
                "p_likes_log": model.pvalues.get("socialProductsLiked_log", np.nan),
                "coef_seniority_centered": model.params.get("seniority_centered", np.nan),
                "p_seniority_centered": model.pvalues.get("seniority_centered", np.nan),
                "coef_seniority_centered_sq": model.params.get(
                    "seniority_centered_sq", np.nan
                ),
                "p_seniority_centered_sq": model.pvalues.get(
                    "seniority_centered_sq", np.nan
                ),
                "coef_is_female": model.params.get("is_female", np.nan),
                "p_is_female": model.pvalues.get("is_female", np.nan),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return results_df


def main():
    project_dir = Path(__file__).resolve().parent.parent
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"
    figures_dir = project_dir / "outputs" / "figures"
    tables_dir = project_dir / "outputs" / "tables"
    reports_dir = project_dir / "outputs" / "reports"

    input_csv = raw_dir / "users_dataset.csv"

    cleaned_log_csv = processed_dir / "log_transformed_features.csv"
    model_ready_csv = processed_dir / "analysis_model_data.csv"
    eda_hist_png = figures_dir / "eda_distributions.png"
    boxplot_png = figures_dir / "log_boxplots.png"
    corr_csv = tables_dir / "correlation_matrix.csv"
    corr_png = figures_dir / "correlation_heatmap.png"
    descriptive_stats_csv = tables_dir / "descriptive_statistics.csv"
    descriptive_stats_md = reports_dir / "descriptive_statistics_table.md"
    ols_txt = reports_dir / "ols_summary.txt"
    ols_coef_csv = tables_dir / "ols_coefficients.csv"
    robust_txt = reports_dir / "ols_summary_robust_hc3.txt"
    robust_coef_csv = tables_dir / "ols_coefficients_robust_hc3.csv"
    diagnostics_txt = reports_dir / "model_diagnostics.txt"
    vif_csv = tables_dir / "vif_results.csv"
    sold_rate_txt = reports_dir / "robustness_sold_rate_summary.txt"
    sold_rate_coef_csv = tables_dir / "robustness_sold_rate_coefficients.csv"
    country_subsample_csv = tables_dir / "robustness_country_subsamples.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Reading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(df.head())

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    required_cols = [
        "productsSold",
        "productsListed",
        "socialNbFollowers",
        "socialProductsLiked",
        "gender",
        "seniority",
        "country",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns are missing: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    print("Step 1/6: Cleaning data and filtering the sample...")
    model_data = df[required_cols].dropna().copy()

    continuous_cols = [
        "productsSold",
        "productsListed",
        "socialNbFollowers",
        "socialProductsLiked",
        "seniority",
    ]
    for col in continuous_cols:
        model_data[col] = winsorize_series(model_data[col], 0.01, 0.99)

    print(f"Sample size after cleaning: {len(model_data)}")

    print("Step 2/6: Building features and saving processed data...")
    log_mapping = {
        "productsSold": "productsSold_log",
        "productsListed": "productsListed_log",
        "socialNbFollowers": "socialNbFollowers_log",
        "socialProductsLiked": "socialProductsLiked_log",
    }
    for source_col, log_col in log_mapping.items():
        model_data[log_col] = np.log1p(model_data[source_col])

    model_data["is_female"] = np.where(model_data["gender"] == "F", 1, 0)
    model_data["seniority_centered"] = (
        model_data["seniority"] - model_data["seniority"].mean()
    )
    model_data["seniority_centered_sq"] = model_data["seniority_centered"] ** 2

    listed_positive_mask = model_data["productsListed"] > 0
    model_data["sold_rate"] = np.where(
        listed_positive_mask,
        model_data["productsSold"] / model_data["productsListed"],
        np.nan,
    )
    sold_rate_upper = model_data["sold_rate"].quantile(0.99)
    model_data["sold_rate"] = model_data["sold_rate"].clip(upper=sold_rate_upper)

    log_data = model_data[list(log_mapping.values())].copy()
    log_data.to_csv(cleaned_log_csv, index=False, encoding="utf-8-sig")
    model_data.to_csv(model_ready_csv, index=False, encoding="utf-8-sig")
    print(f"Saved log-transformed features: {cleaned_log_csv}")
    print(f"Saved model-ready dataset: {model_ready_csv}")

    descriptive_vars = [
        "productsSold",
        "productsListed",
        "socialNbFollowers",
        "socialProductsLiked",
        "seniority",
        "productsSold_log",
        "productsListed_log",
        "socialNbFollowers_log",
        "socialProductsLiked_log",
        "is_female",
    ]
    descriptive_stats = build_descriptive_statistics_table(model_data, descriptive_vars)
    descriptive_stats.to_csv(descriptive_stats_csv, index=False, encoding="utf-8-sig")
    descriptive_stats_md.write_text(
        descriptive_stats_to_text(descriptive_stats),
        encoding="utf-8",
    )
    print(f"Saved descriptive statistics table: {descriptive_stats_csv}")

    print("Step 3/6: Exporting EDA charts and correlation outputs...")
    raw_cols = list(log_mapping.keys())
    log_cols = list(log_mapping.values())

    save_distribution_plots(model_data, raw_cols, log_cols, eda_hist_png)
    save_boxplots(model_data, raw_cols, log_cols, boxplot_png)

    corr_cols = [
        "productsSold_log",
        "productsListed_log",
        "socialNbFollowers_log",
        "socialProductsLiked_log",
        "seniority_centered",
        "seniority_centered_sq",
        "is_female",
    ]
    save_correlation_outputs(model_data, corr_cols, corr_csv, corr_png)

    print("Step 4/6: Fitting baseline OLS models...")
    formula = (
        "productsSold_log ~ productsListed_log + socialNbFollowers_log + "
        "socialProductsLiked_log + seniority_centered + "
        "seniority_centered_sq + is_female"
    )

    ols_model = smf.ols(formula=formula, data=model_data).fit()
    robust_model = smf.ols(formula=formula, data=model_data).fit(cov_type="HC3")

    print(ols_model.summary())
    print(robust_model.summary())
    save_model_outputs(ols_model, ols_txt, ols_coef_csv)
    save_model_outputs(robust_model, robust_txt, robust_coef_csv)

    print("Step 5/6: Running VIF and heteroskedasticity diagnostics...")
    vif_features = [
        "productsListed_log",
        "socialNbFollowers_log",
        "socialProductsLiked_log",
        "seniority_centered",
        "seniority_centered_sq",
        "is_female",
    ]
    vif_table = build_vif_table(model_data, vif_features)
    vif_table.to_csv(vif_csv, index=False, encoding="utf-8-sig")

    bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
    white_exog = sm.add_constant(
        model_data[
            [
                "productsListed_log",
                "socialNbFollowers_log",
                "socialProductsLiked_log",
                "seniority_centered",
                "is_female",
            ]
        ],
        has_constant="add",
    )
    white_test = het_white(ols_model.resid, white_exog)
    save_diagnostic_text(vif_table, bp_test, white_test, diagnostics_txt)
    print(vif_table)

    print("Step 6/6: Running robustness checks...")
    sold_rate_data = model_data.dropna(subset=["sold_rate"]).copy()
    sold_rate_formula = (
        "sold_rate ~ socialNbFollowers_log + socialProductsLiked_log + "
        "seniority_centered + seniority_centered_sq + is_female"
    )
    sold_rate_model = smf.ols(formula=sold_rate_formula, data=sold_rate_data).fit(
        cov_type="HC3"
    )
    print(sold_rate_model.summary())
    save_model_outputs(sold_rate_model, sold_rate_txt, sold_rate_coef_csv)

    subsample_results = run_country_subsample_models(
        model_data=model_data,
        formula=formula,
        output_csv_path=country_subsample_csv,
        top_n=3,
    )
    print(subsample_results)

    print("Analysis pipeline completed.")


if __name__ == "__main__":
    main()
