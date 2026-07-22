"""
Logica di calcolo estratta dal notebook Modelli.ipynb.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeClassifier

Cf_list = [10000, 20000, 50000]
MTTF_list = [0.5, 1, 2]
beta_list = [1.5, 2, 2.5, 3, 3.5]
CSystPdM_list = [1000, 2000, 4000, 5000, 10000, 25000]
alfa_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Cinter_list = [1000, 2000, 4000, 5000, 10000]
AN_list = [1 - (1 / 880), 1 - (1 / 1760), 1 - (1 / 3520)]
fsc_list = [1]
Nh_list = [1760]
r_list = [0.99]

detectability_scenarios = [
    {"detectability": "low", "H": 0.65, "F": 0.30},
    {"detectability": "medium", "H": 0.85, "F": 0.10},
    {"detectability": "high", "H": 0.95, "F": 0.05},
]

FEATURES = ["Cinter", "CSystPdM", "Beta", "Alfa"]
TARGET = "Strategia_Ottimale"

SCENARI_PREDICTABILITY = {
    "HIGH": {"H": 0.95, "F": 0.05},
    "MEDIUM": {"H": 0.85, "F": 0.10},
    "LOW": {"H": 0.65, "F": 0.30},
}
SCENARI_SEVERITY = {"HIGH": 50000, "MEDIUM": 20000, "LOW": 10000}
SCENARI_OCCURRENCE = {"HIGH": 0.5, "MEDIUM": 1.0, "LOW": 2.0}


def calcola_df_guasto():
    combinazioni_essenziali = list(itertools.product(MTTF_list, Cf_list))
    risultati_guasto = []
    for mttf, cf in combinazioni_essenziali:
        costo_guasto = cf / mttf
        risultati_guasto.append(
            {"MTTF": mttf, "Cf": cf, "Costo_CORRECTIVE": round(costo_guasto, 4)}
        )
    return pd.DataFrame(risultati_guasto)


def calcola_df_preventiva(weibull_csv_path):
    df_raw = pd.read_csv(weibull_csv_path, header=None, index_col=0)
    beta_riferimento = df_raw.loc["beta"].values.astype(float)
    rapporto_riferimento = df_raw.loc["MTTF/theta"].values.astype(float)

    def get_theta(beta_val, mttf_val):
        rapporto = np.interp(beta_val, beta_riferimento, rapporto_riferimento)
        return mttf_val / rapporto

    def costo_obiettivo(tp_val, b, th, cf, cint, csys, alfa, dettagli=False):
        if tp_val <= 0:
            return np.inf
        rtp = np.exp(-((tp_val / th) ** b))
        area, _ = quad(lambda s: np.exp(-((s / th) ** b)), 0, tp_val)
        if area == 0:
            return np.inf
        c_failure = (cf * (1 - rtp)) / area
        c_preventivo = (cint * rtp) / area
        c_monitoraggio = alfa * csys
        costo_totale = c_failure + c_preventivo + c_monitoraggio
        if dettagli:
            return costo_totale, c_failure, c_preventivo, c_monitoraggio
        return costo_totale

    risultati_totali = []
    combinazioni = list(
        itertools.product(beta_list, MTTF_list, Cf_list, Cinter_list, CSystPdM_list, alfa_list)
    )

    for b, mttf, cf, cint, csys, alfa in combinazioni:
        th = get_theta(b, mttf)
        res = minimize_scalar(
            costo_obiettivo,
            bounds=(1, mttf * 5),
            args=(b, th, cf, cint, csys, alfa, False),
            method="bounded",
        )
        tot, fail, prev, mon = costo_obiettivo(res.x, b, th, cf, cint, csys, alfa, dettagli=True)
        risultati_totali.append(
            {
                "Beta": b,
                "MTTF": mttf,
                "Cf": cf,
                "Cinter": cint,
                "CSystPdM": csys,
                "Alfa": alfa,
                "Theta": th,
                "tp_Ottimo": round(res.x, 2),
                "Costo_Failure_prev": round(fail, 4),
                "Costo_Prevenzione_prev": round(prev, 4),
                "Costo_Monitoraggio_prev": round(mon, 4),
                "Costo_PREVENTIVE_Totale": round(res.fun, 4),
            }
        )

    return pd.DataFrame(risultati_totali)


def calcola_df_predittiva():
    combinazioni_pdm_base = list(
        itertools.product(Cf_list, Cinter_list, CSystPdM_list, MTTF_list, AN_list, fsc_list, Nh_list, r_list)
    )

    risultati_predittiva = []
    for cf, cint, csys, mttf, an, fsc, nh, r in combinazioni_pdm_base:
        for det in detectability_scenarios:
            h = det["H"]
            f = det["F"]

            t1 = csys
            t2 = cint * f * an * fsc * nh * (1 - r)
            t3 = (cint * h) / mttf
            t4 = (cf * (1 - h)) / mttf

            costo_totale_pred = t1 + t2 + t3 + t4

            risultati_predittiva.append(
                {
                    "Cf": cf,
                    "Cinter": cint,
                    "CSystPdM": csys,
                    "MTTF": mttf,
                    "F": f,
                    "H": h,
                    "AN": an,
                    "fsc": fsc,
                    "Nh": nh,
                    "r": r,
                    "Costo_falsi_positivi": t1,
                    "Costo_falsi_negativi": t4,
                    "Costo_veri_positivi": t3,
                    "Costo_PREDICTIVE_Totale": round(costo_totale_pred, 4),
                }
            )

    return pd.DataFrame(risultati_predittiva)


def calcola_df_confronto_globale(df_guasto, df_preventiva, df_predittiva):
    df_confronto = pd.merge(
        df_predittiva, df_preventiva, on=["Cf", "Cinter", "CSystPdM", "MTTF"], how="inner"
    )
    df_confronto = pd.merge(df_confronto, df_guasto, on=["Cf", "MTTF"], how="inner")

    colonne_costi = ["Costo_CORRECTIVE", "Costo_PREVENTIVE_Totale", "Costo_PREDICTIVE_Totale"]
    df_confronto["Strategia_Ottimale"] = df_confronto[colonne_costi].idxmin(axis=1)
    df_confronto["Strategia_Ottimale"] = (
        df_confronto["Strategia_Ottimale"].str.replace("Costo_", "").str.replace("_Totale", "")
    )
    return df_confronto


COST_COLUMNS = {
    "CORRECTIVE": "Costo_CORRECTIVE",
    "PREDICTIVE": "Costo_PREDICTIVE_Totale",
    "PREVENTIVE": "Costo_PREVENTIVE_Totale",
}


def calcola_leaf_stats(clf, df_scenario):
    """Per ogni foglia dell'albero, calcola accuracy ed 'expected extra cost':
    il costo medio extra (in euro e in %) pagato nei casi storici in cui la
    foglia ha sbagliato la previsione, pesato per la probabilita' di sbagliare
    (inaccuracy) di quella foglia. Stessa logica del file Leaf_Summary_Statistics.
    """
    X = df_scenario[FEATURES]
    y_true = df_scenario[TARGET]
    y_pred = clf.predict(X)
    leaf_ids = clf.apply(X)

    d = df_scenario.copy()
    d["_pred"] = y_pred
    d["_leaf"] = leaf_ids
    d["_correct"] = d["_pred"] == d[TARGET]

    extra_cost = np.zeros(len(d))
    extra_cost_pct = np.zeros(len(d))
    savings = np.zeros(len(d))
    savings_pct = np.zeros(len(d))
    pred_vals = d["_pred"].to_numpy()
    true_vals = d[TARGET].to_numpy()
    correct_vals = d["_correct"].to_numpy()

    costi = {name: d[col].to_numpy() for name, col in COST_COLUMNS.items()}
    c_corr_arr = d["Costo_CORRECTIVE"].to_numpy()
    for i in range(len(d)):
        if correct_vals[i]:
            # predizione corretta: risparmio rispetto al costo "a guasto" (baseline)
            cost_model_val = costi[pred_vals[i]][i]
            s = c_corr_arr[i] - cost_model_val
            savings[i] = s
            savings_pct[i] = s / c_corr_arr[i] if c_corr_arr[i] != 0 else 0
        else:
            # predizione errata: costo extra rispetto alla strategia veramente ottimale
            cost_pred_val = costi[pred_vals[i]][i]
            cost_true_val = costi[true_vals[i]][i]
            ec = cost_pred_val - cost_true_val
            extra_cost[i] = ec
            extra_cost_pct[i] = ec / cost_true_val if cost_true_val != 0 else 0

    d["_extra_cost"] = extra_cost
    d["_extra_cost_pct"] = extra_cost_pct
    d["_savings"] = savings
    d["_savings_pct"] = savings_pct

    stats = {}
    for leaf_id, g in d.groupby("_leaf"):
        n = len(g)
        n_correct = int(g["_correct"].sum())
        accuracy = n_correct / n
        inaccuracy = 1 - accuracy

        correct = g.loc[g["_correct"]]
        if len(correct) > 0:
            saving_mean = correct["_savings"].mean()
            saving_mean_pct = correct["_savings_pct"].mean()
        else:
            saving_mean = 0.0
            saving_mean_pct = 0.0

        wrong = g.loc[~g["_correct"]]
        if len(wrong) > 0:
            cost_mean = wrong["_extra_cost"].mean()
            cost_mean_pct = wrong["_extra_cost_pct"].mean()
        else:
            cost_mean = 0.0
            cost_mean_pct = 0.0

        stats[int(leaf_id)] = {
            "n_samples": n,
            "accuracy": accuracy,
            "inaccuracy": inaccuracy,
            "saving_expected": saving_mean * accuracy,
            "saving_expected_pct": saving_mean_pct * accuracy,
            "cost_expected": cost_mean * inaccuracy,
            "cost_expected_pct": cost_mean_pct * inaccuracy,
        }
    return stats


def addestra_modelli(df_confronto_globale):
    models = {}
    accuracies = {}
    leaf_stats = {}

    for name_p, p_val in SCENARI_PREDICTABILITY.items():
        for name_s, s_val in SCENARI_SEVERITY.items():
            for name_o, o_val in SCENARI_OCCURRENCE.items():
                df_scenario = df_confronto_globale[
                    (df_confronto_globale["H"] == p_val["H"])
                    & (df_confronto_globale["F"] == p_val["F"])
                    & (df_confronto_globale["Cf"] == s_val)
                    & (df_confronto_globale["MTTF"] == o_val)
                ].copy()

                if df_scenario.empty:
                    continue

                X = df_scenario[FEATURES]
                y = df_scenario[TARGET]

                clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                clf.fit(X, y)
                key = (name_p, name_s, name_o)
                models[key] = clf
                accuracies[key] = clf.score(X, y)
                leaf_stats[key] = calcola_leaf_stats(clf, df_scenario)

    return models, accuracies, leaf_stats


def query_tree(models, leaf_stats, p, s, o, cinter, csystpdm, beta, alfa):
    key = (p, s, o)
    if key not in models:
        return {"error": "Scenario non trovato"}

    clf = models[key]
    X_new = pd.DataFrame(
        [{"Cinter": cinter, "CSystPdM": csystpdm, "Beta": beta, "Alfa": alfa}]
    )
    pred = clf.predict(X_new)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(X_new)[0]))
    leaf_id = int(clf.apply(X_new)[0])
    stats = leaf_stats.get(key, {}).get(leaf_id, {})

    return {
        "scenario": key,
        "strategy": pred,
        "proba": proba,
        "leaf_id": leaf_id,
        "n_samples": stats.get("n_samples"),
        "accuracy": stats.get("accuracy"),
        "inaccuracy": stats.get("inaccuracy"),
        "saving_expected": stats.get("saving_expected"),
        "saving_expected_pct": stats.get("saving_expected_pct"),
        "cost_expected": stats.get("cost_expected"),
        "cost_expected_pct": stats.get("cost_expected_pct"),
    }


def build_all(weibull_csv_path):
    df_guasto = calcola_df_guasto()
    df_preventiva = calcola_df_preventiva(weibull_csv_path)
    df_predittiva = calcola_df_predittiva()
    df_confronto = calcola_df_confronto_globale(df_guasto, df_preventiva, df_predittiva)
    models, accuracies, leaf_stats = addestra_modelli(df_confronto)
    return {
        "df_guasto": df_guasto,
        "df_preventiva": df_preventiva,
        "df_predittiva": df_predittiva,
        "df_confronto": df_confronto,
        "models": models,
        "accuracies": accuracies,
        "leaf_stats": leaf_stats,
    }
