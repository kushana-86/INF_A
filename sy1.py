from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_DIR = Path(__file__).resolve().parent
ENTER_P = 0.05
REMOVE_P = 0.10
PLOT_FILE = "自动筛选高相关变量散点图.png"
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


def get_variable_labels() -> dict[str, str]:
    return {
        "y": "全国城镇调查失业率",
        "X1": "31个大城市城镇调查失业率",
        "X2": "全国城镇本地户籍劳动力失业率",
        "X3": "全国城镇外来户籍劳动力失业率",
        "X4": "全国城镇16—24岁劳动力失业率",
        "X5": "全国城镇25—29岁劳动力失业率",
        "X6": "全国城镇30—59岁劳动力失业率",
        "X7": "企业就业人员周平均工作时间",
    }


def fit_model(df: pd.DataFrame, response: str, features: list[str]):
    x = sm.add_constant(df[features])
    return sm.OLS(df[response], x).fit()


def calculate_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = sm.add_constant(df[features])
    vif_rows = []
    for i, col in enumerate(x.columns):
        vif_rows.append({"变量": col, "VIF": variance_inflation_factor(x.values, i)})
    return pd.DataFrame(vif_rows)


def select_high_correlation_pair(data: pd.DataFrame, candidates: list[str]):
    corr_matrix = data[candidates].corr()
    best_abs_corr = -1.0
    best_pair = None

    for i, var1 in enumerate(candidates):
        for var2 in candidates[i + 1 :]:
            pair_data = data[[var1, var2]].dropna()
            if len(pair_data) < 3:
                continue
            corr_value = pair_data[var1].corr(pair_data[var2])
            if pd.isna(corr_value):
                continue
            if abs(corr_value) > best_abs_corr:
                best_abs_corr = abs(corr_value)
                best_pair = (var1, var2, corr_value, len(pair_data))

    if best_pair is None:
        raise ValueError("未找到可用于相关分析的变量对。")

    return best_pair, corr_matrix


def compute_partial_corr(data: pd.DataFrame, x_var: str, y_var: str, control_var: str):
    partial_data = data[[x_var, y_var, control_var]].dropna().copy()
    x_model = sm.OLS(partial_data[x_var], sm.add_constant(partial_data[[control_var]])).fit()
    y_model = sm.OLS(partial_data[y_var], sm.add_constant(partial_data[[control_var]])).fit()

    r, p_value = pearsonr(x_model.resid, y_model.resid)
    return partial_data, r, p_value


def run_association_analyses(data: pd.DataFrame, labels: dict[str, str]):
    candidates = ["y", "X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    best_pair, corr_matrix = select_high_correlation_pair(data, candidates)
    x_var, y_var, corr_value, sample_size = best_pair

    remaining = [col for col in candidates if col not in (x_var, y_var)]
    control_var = max(
        remaining,
        key=lambda col: abs(data[[x_var, col]].dropna()[x_var].corr(data[[x_var, col]].dropna()[col]))
        if len(data[[x_var, col]].dropna()) >= 3
        else -1,
    )

    pair_data = data[["month", x_var, y_var]].dropna().copy()
    partial_data, partial_r, partial_p = compute_partial_corr(data, x_var, y_var, control_var)

    distance_input = pair_data[[x_var, y_var]]
    distance_matrix = pd.DataFrame(
        squareform(pdist(distance_input, metric="euclidean")),
        index=pair_data["month"],
        columns=pair_data["month"],
    )

    pearson_r, pearson_p = pearsonr(pair_data[x_var], pair_data[y_var])

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 6))
    plt.scatter(pair_data[x_var], pair_data[y_var], color="#2f6b4f")
    plt.xlabel(f"{x_var}: {labels[x_var]}")
    plt.ylabel(f"{y_var}: {labels[y_var]}")
    plt.title("自动筛选的高相关变量散点图")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(DATA_DIR / PLOT_FILE, dpi=300)
    plt.close()

    corr_summary = pd.DataFrame(
        [
            {
                "x变量": x_var,
                "x变量含义": labels[x_var],
                "y变量": y_var,
                "y变量含义": labels[y_var],
                "样本量": sample_size,
                "Pearson相关系数": pearson_r,
                "相关检验p值": pearson_p,
            }
        ]
    )

    partial_summary = pd.DataFrame(
        [
            {
                "x变量": x_var,
                "y变量": y_var,
                "控制变量": control_var,
                "控制变量含义": labels[control_var],
                "样本量": len(partial_data),
                "一阶偏相关系数": partial_r,
                "偏相关检验p值": partial_p,
            }
        ]
    )

    return {
        "corr_matrix": corr_matrix,
        "pair_data": pair_data,
        "corr_summary": corr_summary,
        "partial_summary": partial_summary,
        "distance_matrix": distance_matrix,
        "x_var": x_var,
        "y_var": y_var,
        "control_var": control_var,
        "corr_value": corr_value,
    }


def run_simple_regression(data: pd.DataFrame):
    simple_data = data[["y", "X7"]].dropna().copy()
    simple_model = fit_model(simple_data.rename(columns={"X7": "target"}), "target", ["y"])

    summary_df = pd.DataFrame(
        [
            {
                "样本量": len(simple_data),
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
                model = fit_model(df[[response] + included + [new_col]], response, included + [new_col])
                new_pvalues[new_col] = model.pvalues[new_col]

            best_feature = new_pvalues.idxmin()
            best_pvalue = float(new_pvalues.min())
            if best_pvalue < threshold_in:
                included.append(best_feature)
                model = fit_model(df[[response] + included], response, included)
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
            model = fit_model(df[[response] + included], response, included)
            pvalues = model.pvalues.drop("const")
            worst_feature = pvalues.idxmax()
            worst_pvalue = float(pvalues.max())
            if worst_pvalue > threshold_out:
                included.remove(worst_feature)
                model = fit_model(df[[response] + included], response, included) if included else None
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

    final_model = fit_model(df[[response] + included], response, included)
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
        model = fit_model(df[["y"] + features], "y", features)
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
    labels: dict[str, str],
    assoc_results: dict,
    simple_summary: pd.DataFrame,
    simple_model,
    modeling_data: pd.DataFrame,
    step_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
):
    corr_series = modeling_data.corr(numeric_only=True)["y"].drop("y").sort_values(ascending=False)
    best_model = comparison_df.sort_values("AIC").iloc[0]
    simple_row = simple_summary.iloc[0]
    x_var = assoc_results["x_var"]
    y_var = assoc_results["y_var"]
    control_var = assoc_results["control_var"]
    corr_row = assoc_results["corr_summary"].iloc[0]
    partial_row = assoc_results["partial_summary"].iloc[0]

    lines = [
        "# 回归分析结论",
        "",
        f"- 数据文件：`{excel_name}`",
        f"- 原始月份数：{len(data)}",
        f"- 多元回归建模样本量：{len(modeling_data)}",
        "",
        "## 相关分析、偏相关分析与距离分析",
        "",
        f"- 自动筛选出的高相关变量对为：`{x_var}`（{labels[x_var]}）与 `{y_var}`（{labels[y_var]}）。",
        f"- Pearson 相关系数 r = {corr_row['Pearson相关系数']:.4f}，p = {corr_row['相关检验p值']:.4f}。",
        f"- 一阶偏相关控制变量为 `{control_var}`（{labels[control_var]}），偏相关系数 = {partial_row['一阶偏相关系数']:.4f}，p = {partial_row['偏相关检验p值']:.4f}。",
        "- 距离分析使用上述两变量构造欧氏距离矩阵，结果已写入 Excel。",
        "",
        "## 一元回归",
        "",
        "- 自变量：`y`（全国城镇调查失业率）",
        "- 因变量：`X7`（企业就业人员周平均工作时间）",
        f"- 一元回归方程：`{model_equation(simple_model, response_name='X7')}`",
        f"- 调整R² = {simple_row['调整R²']:.4f}",
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
            "- 相关分析部分不再手工指定 x 和 y，而是自动选择表中相关性最高的一对变量进行分析。",
            "- 逐步回归部分仍以全国城镇调查失业率 y 为因变量，便于和你的实验主题保持一致。",
            "- 最终模型解释力较强，但样本量有限，且未单独处理时间序列特征，结论更适合课程分析与结构解释。",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    excel_path = find_excel_file()
    labels = get_variable_labels()
    data = load_and_prepare_data(excel_path)

    assoc_results = run_association_analyses(data, labels)
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
        assoc_results["corr_matrix"].to_excel(writer, sheet_name="相关系数矩阵")
        assoc_results["pair_data"].to_excel(writer, sheet_name="高相关变量样本", index=False)
        assoc_results["corr_summary"].to_excel(writer, sheet_name="相关分析汇总", index=False)
        assoc_results["partial_summary"].to_excel(writer, sheet_name="偏相关分析汇总", index=False)
        assoc_results["distance_matrix"].to_excel(writer, sheet_name="距离矩阵")
        simple_data.to_excel(writer, sheet_name="一元回归样本", index=False)
        simple_summary.to_excel(writer, sheet_name="一元回归汇总", index=False)
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
        labels,
        assoc_results,
        simple_summary,
        simple_model,
        modeling_data,
        step_df,
        comparison_df,
    )

    print("代码已更新并运行完成。")
    print(f"自动选中的高相关变量对: {assoc_results['x_var']} 和 {assoc_results['y_var']}")
    print(f"对应相关系数: {assoc_results['corr_value']:.4f}")
    print(f"偏相关控制变量: {assoc_results['control_var']}")
    print(f"多元回归最终入选变量: {', '.join(selected_vars)}")
    print("最终模型摘要:")
    print(final_model.summary())
    print(f"结果文件已保存: {output_excel.name}")
    print(f"结论文件已保存: {report_path.name}")
    print(f"散点图已保存: {PLOT_FILE}")


if __name__ == "__main__":
    main()
