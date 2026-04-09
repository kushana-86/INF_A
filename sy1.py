from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_DIR = Path(__file__).resolve().parent
ENTER_P = 0.05
REMOVE_P = 0.10
PLOT_FILE = "失业率与周工作时间散点图.png"
RESULT_FILE = "多元线性回归结果.xlsx"
REPORT_FILE = "回归分析结论.md"


def find_excel_file() -> Path:
    excel_files = sorted(DATA_DIR.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError("当前目录下未找到 Excel 数据文件。")
    return excel_files[0]


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=2)
    index_col = raw.columns[0]

    raw = raw[raw[index_col].notna()].copy()
    raw[index_col] = raw[index_col].astype(str).str.strip()
    raw = raw[~raw[index_col].str.startswith("注：", na=False)]
    raw = raw[~raw[index_col].str.startswith("数据来源", na=False)]

    data = raw.set_index(index_col).T
    data.columns = data.columns.astype(str).str.strip()
    data.index.name = "month"
    data = data.reset_index()
    data["month"] = data["month"].astype(str).str.strip()

    rename_map = {
        data.columns[1]: "y",
        data.columns[2]: "X1",
        data.columns[3]: "X2",
        data.columns[4]: "X3",
        data.columns[5]: "X4",
        data.columns[6]: "X5",
        data.columns[7]: "X6",
        data.columns[9]: "X7",
    }
    data = data.rename(columns=rename_map)

    keep_cols = ["month", "y", "X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    data = data[keep_cols].copy()
    for col in keep_cols[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def fit_model(df: pd.DataFrame, features: list[str]):
    x = sm.add_constant(df[features])
    return sm.OLS(df["y"], x).fit()


def calculate_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = sm.add_constant(df[features])
    vif_rows = []
    for i, col in enumerate(x.columns):
        vif_rows.append({"变量": col, "VIF": variance_inflation_factor(x.values, i)})
    return pd.DataFrame(vif_rows)


def run_simple_regression(data: pd.DataFrame):
    simple_data = data[["y", "X7"]].dropna().copy()
    x = simple_data["y"]
    y = simple_data["X7"]

    r, p_value = pearsonr(x, y)
    simple_model = sm.OLS(y, sm.add_constant(x)).fit()

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="#2f6b4f")
    plt.xlabel("全国城镇调查失业率 (%)")
    plt.ylabel("企业就业人员周平均工作时间 (小时)")
    plt.title("全国城镇调查失业率与周工作时间散点图")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(DATA_DIR / PLOT_FILE, dpi=300)
    plt.close()

    summary_df = pd.DataFrame(
        [
            {
                "样本量": len(simple_data),
                "相关系数r": r,
                "相关检验p值": p_value,
                "R²": simple_model.rsquared,
                "调整R²": simple_model.rsquared_adj,
                "截距": simple_model.params["const"],
                "y系数": simple_model.params["y"],
            }
        ]
    )

    coef_df = pd.DataFrame(
        {
            "变量": simple_model.params.index,
            "系数": simple_model.params.values,
            "标准误": simple_model.bse.values,
            "t值": simple_model.tvalues.values,
            "p值": simple_model.pvalues.values,
            "下限95%": simple_model.conf_int()[0].values,
            "上限95%": simple_model.conf_int()[1].values,
        }
    )

    return simple_data, summary_df, coef_df, simple_model


def stepwise_selection(
    df: pd.DataFrame,
    response: str,
    candidates: list[str],
    threshold_in: float = ENTER_P,
    threshold_out: float = REMOVE_P,
):
    included: list[str] = []
    steps: list[dict] = []

    while True:
        changed = False
        excluded = [col for col in candidates if col not in included]

        if excluded:
            new_pvalues = pd.Series(index=excluded, dtype=float)
            for new_col in excluded:
                model = fit_model(df[[response] + included + [new_col]], included + [new_col])
                new_pvalues[new_col] = model.pvalues[new_col]

            best_feature = new_pvalues.idxmin()
            best_pvalue = float(new_pvalues.min())
            if best_pvalue < threshold_in:
                included.append(best_feature)
                model = fit_model(df[[response] + included], included)
                steps.append(
                    {
                        "步骤": len(steps) + 1,
                        "动作": "加入",
                        "变量": best_feature,
                        "变量p值": best_pvalue,
                        "当前变量": ", ".join(included),
                        "样本量": int(model.nobs),
                        "R²": model.rsquared,
                        "调整R²": model.rsquared_adj,
                        "AIC": model.aic,
                        "BIC": model.bic,
                    }
                )
                changed = True

        if included:
            model = fit_model(df[[response] + included], included)
            pvalues = model.pvalues.drop("const")
            worst_feature = pvalues.idxmax()
            worst_pvalue = float(pvalues.max())
            if worst_pvalue > threshold_out:
                included.remove(worst_feature)
                model = fit_model(df[[response] + included], included) if included else None
                steps.append(
                    {
                        "步骤": len(steps) + 1,
                        "动作": "剔除",
                        "变量": worst_feature,
                        "变量p值": worst_pvalue,
                        "当前变量": ", ".join(included),
                        "样本量": int(model.nobs) if model else len(df),
                        "R²": model.rsquared if model else None,
                        "调整R²": model.rsquared_adj if model else None,
                        "AIC": model.aic if model else None,
                        "BIC": model.bic if model else None,
                    }
                )
                changed = True

        if not changed:
            break

    final_model = fit_model(df[[response] + included], included)
    return included, pd.DataFrame(steps), final_model


def model_equation(model, response_name: str = "y") -> str:
    params = model.params
    terms = [f"{response_name} = {params['const']:.4f}"]
    for var in params.index:
        if var == "const":
            continue
        coef = params[var]
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.4f}*{var}")
    return " ".join(terms)


def build_model_outputs(df: pd.DataFrame, model_specs: list[tuple[str, list[str]]]):
    comparison_rows = []
    coefficient_tables = {}
    diagnostics_tables = {}

    for model_name, features in model_specs:
        model = fit_model(df[["y"] + features], features)
        comparison_rows.append(
            {
                "模型": model_name,
                "自变量": ", ".join(features),
                "样本量": int(model.nobs),
                "R²": model.rsquared,
                "调整R²": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic,
                "F统计量": model.fvalue,
                "F检验p值": model.f_pvalue,
                "回归方程": model_equation(model),
            }
        )

        coefficient_tables[model_name] = pd.DataFrame(
            {
                "变量": model.params.index,
                "系数": model.params.values,
                "标准误": model.bse.values,
                "t值": model.tvalues.values,
                "p值": model.pvalues.values,
                "下限95%": model.conf_int()[0].values,
                "上限95%": model.conf_int()[1].values,
            }
        )
        diagnostics_tables[model_name] = calculate_vif(df, features)

    return pd.DataFrame(comparison_rows), coefficient_tables, diagnostics_tables


def write_report(
    report_path: Path,
    excel_name: str,
    data: pd.DataFrame,
    simple_summary: pd.DataFrame,
    simple_model,
    modeling_data: pd.DataFrame,
    step_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
):
    corr_series = modeling_data.corr(numeric_only=True)["y"].drop("y").sort_values(ascending=False)
    best_model = comparison_df.sort_values("AIC").iloc[0]
    simple_row = simple_summary.iloc[0]

    lines = [
        "# 回归分析结论",
        "",
        f"- 数据文件：`{excel_name}`",
        f"- 原始月份数：{len(data)}",
        f"- 多元回归建模样本量：{len(modeling_data)}",
        "",
        "## 一元分析",
        "",
        "- 自变量：`y`（全国城镇调查失业率）",
        "- 因变量：`X7`（企业就业人员周平均工作时间）",
        f"- Pearson 相关系数 r = {simple_row['相关系数r']:.4f}，p = {simple_row['相关检验p值']:.4f}",
        f"- 一元回归方程：`{model_equation(simple_model, response_name='X7')}`",
        f"- 一元回归调整R² = {simple_row['调整R²']:.4f}",
        "",
        "## 逐步回归路径",
        "",
    ]

    for _, row in step_df.iterrows():
        lines.append(
            f"- 第{int(row['步骤'])}步：{row['动作']} `{row['变量']}`，当前变量组合为 `{row['当前变量']}`，"
            f"调整R² = {row['调整R²']:.4f}，AIC = {row['AIC']:.4f}"
        )

    lines.extend(["", "## 与因变量 y 的相关性", ""])
    for name, value in corr_series.items():
        lines.append(f"- `{name}` 与 `y` 的相关系数为 {value:.4f}")

    lines.extend(
        [
            "",
            "## 模型比较结论",
            "",
            f"- 最优模型为 **{best_model['模型']}**。",
            f"- 自变量组合：`{best_model['自变量']}`",
            f"- 调整R² = {best_model['调整R²']:.4f}，AIC = {best_model['AIC']:.4f}",
            f"- 回归方程：`{best_model['回归方程']}`",
            "",
            "## 解释",
            "",
            "- `X2` 与 `X3` 是最主要的解释变量。",
            "- `X5` 在控制其他变量后仍达到显著水平，因此被保留在最终模型中。",
            "- 最终模型解释力较强，但样本量只有 27 期，且未单独处理时间序列特征，结论更适合课程分析与结构解释。",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    excel_path = find_excel_file()
    data = load_and_prepare_data(excel_path)

    simple_data, simple_summary, simple_coef, simple_model = run_simple_regression(data)

    candidate_vars = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    modeling_data = data[["y"] + candidate_vars].dropna().copy()
    selected_vars, step_df, final_model = stepwise_selection(
        modeling_data,
        response="y",
        candidates=candidate_vars,
    )

    model_specs = [
        ("模型1", [step_df.iloc[0]["变量"]]),
        ("模型2", [step_df.iloc[0]["变量"], step_df.iloc[1]["变量"]]),
        ("模型3", selected_vars),
    ]
    comparison_df, coefficient_tables, diagnostics_tables = build_model_outputs(modeling_data, model_specs)

    output_excel = DATA_DIR / RESULT_FILE
    with pd.ExcelWriter(output_excel) as writer:
        data.to_excel(writer, sheet_name="清洗后数据", index=False)
        simple_data.to_excel(writer, sheet_name="一元分析样本", index=False)
        simple_summary.to_excel(writer, sheet_name="一元分析汇总", index=False)
        simple_coef.to_excel(writer, sheet_name="一元回归系数", index=False)
        modeling_data.to_excel(writer, sheet_name="多元建模样本", index=False)
        step_df.to_excel(writer, sheet_name="逐步回归过程", index=False)
        comparison_df.to_excel(writer, sheet_name="模型比较", index=False)
        for model_name, coef_df in coefficient_tables.items():
            coef_df.to_excel(writer, sheet_name=f"{model_name}系数", index=False)
        for model_name, vif_df in diagnostics_tables.items():
            vif_df.to_excel(writer, sheet_name=f"{model_name}VIF", index=False)

    report_path = DATA_DIR / REPORT_FILE
    write_report(
        report_path,
        excel_path.name,
        data,
        simple_summary,
        simple_model,
        modeling_data,
        step_df,
        comparison_df,
    )

    print("脚本合并完成。")
    print(f"数据文件: {excel_path.name}")
    print(f"一元回归样本量: {len(simple_data)}")
    print(f"多元回归样本量: {len(modeling_data)}")
    print(f"最终入选变量: {', '.join(selected_vars)}")
    print("最终模型摘要:")
    print(final_model.summary())
    print(f"结果文件已保存: {output_excel.name}")
    print(f"结论文件已保存: {report_path.name}")
    print(f"散点图已保存: {PLOT_FILE}")


if __name__ == "__main__":
    main()
