"""
Subsidising Green Hydrogen: domestic hydrogen market model (web version).

Australia's domestic hydrogen market on its own. The electricity price is an input. Each case is a linear programme
solved with HiGHS (scipy.optimize.linprog); per-kg subsidies follow the paper's common fiscal rule.

Runs three ways, all from the same files:
  * GitHub Pages: index.html loads this app in the visitor's browser (stlite / Pyodide), no server needed.
  * Streamlit Community Cloud: point it at this repo, main file app.py.
  * Your own computer:  python -m pip install -r requirements.txt
                        python -m streamlit run app.py
"""
from dataclasses import asdict, fields

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from h2_model import Calibration, POLICIES, PER_KG, GRANTS, run_case, summary, paths

st.set_page_config(page_title="Domestic hydrogen market model", layout="wide")

MODEL = "domestic"
DEF = Calibration()

INTRO = """
Australia's domestic hydrogen market on its own. Incumbent steam methane reforming (SMR) and grid-connected
electrolysers compete to meet **inelastic domestic demand** (0.5 Mt in 2026 rising to 3.4 Mt in 2050). SMR pays for gas
and, under the **Safeguard Mechanism**, for emissions above a declining baseline. Its existing plants have sunk capital,
so electrolysis must undercut SMR's *running* cost to displace it. Electrolysers buy electricity at the **price you
enter** and choose capacity, subject to a growth limit. Nothing is exported.

Each instrument has the same **A\\$6.7 bn** envelope over FY2027-28 to FY2039-40. Per-kg instruments pay their announced
rate until the money runs out. The tax offset can draw on the whole envelope from the first year; the others receive
A\\$515 M a year, and unspent money carries forward.
"""

# inputs shown: (group, field, label, step, format)
INPUTS = [
    ("General and domestic demand", "discount_rate", "Real discount rate", 0.005, "%.3f"),
    ("General and domestic demand", "demand_2026", "Domestic H2 demand 2026 (kt/yr)", 50.0, "%.0f"),
    ("General and domestic demand", "demand_2050", "Domestic H2 demand 2050 (kt/yr)", 100.0, "%.0f"),
    ("SMR and the Safeguard Mechanism", "gas_price_2026", "Gas price 2026 (A$/GJ)", 0.1, "%.2f"),
    ("SMR and the Safeguard Mechanism", "gas_price_2027", "Gas price 2027 (A$/GJ)", 0.1, "%.2f"),
    ("SMR and the Safeguard Mechanism", "gas_price_2032", "Gas price 2032 onwards (A$/GJ)", 0.1, "%.2f"),
    ("SMR and the Safeguard Mechanism", "smr_gas_use", "SMR gas use (GJ/kg H2)", 0.005, "%.3f"),
    ("SMR and the Safeguard Mechanism", "smr_capex", "SMR capex (A$/kW gas input)", 50.0, "%.0f"),
    ("SMR and the Safeguard Mechanism", "smr_cost_scale", "SMR running-cost scale (fitted)", 0.01, "%.2f"),
    ("SMR and the Safeguard Mechanism", "smr_spread", "SMR cost spread across plants (A$/kg)", 0.05, "%.2f"),
    ("SMR and the Safeguard Mechanism", "emissions_intensity", "Emissions intensity (kgCO2/kg H2)", 0.1, "%.1f"),
    ("SMR and the Safeguard Mechanism", "accu_2028", "Carbon credit price to 2028 (A$/t)", 1.0, "%.0f"),
    ("SMR and the Safeguard Mechanism", "accu_2035", "Carbon credit price 2035 (A$/t)", 1.0, "%.0f"),
    ("SMR and the Safeguard Mechanism", "accu_growth_after_2035", "Credit price growth after 2035", 0.005, "%.3f"),
    ("Electrolysers", "p2h_capex_2026", "Electrolyser capex 2026 (A$/kW)", 25.0, "%.0f"),
    ("Electrolysers", "p2h_capex_2035", "Electrolyser capex 2035 (A$/kW)", 25.0, "%.1f"),
    ("Electrolysers", "p2h_capex_decline_to_2040", "Capex decline 2035-40 (/yr)", 0.005, "%.3f"),
    ("Electrolysers", "p2h_capex_decline_after_2040", "Capex decline after 2040 (/yr)", 0.001, "%.3f"),
    ("Electrolysers", "hurdle_2026", "Hurdle rate 2026", 0.005, "%.3f"),
    ("Electrolysers", "hurdle_2035", "Hurdle rate 2035 onwards", 0.005, "%.3f"),
    ("Electrolysers", "eff_2026", "Efficiency 2026 (LHV)", 0.005, "%.4f"),
    ("Electrolysers", "eff_2030", "Efficiency 2030 (LHV)", 0.005, "%.3f"),
    ("Electrolysers", "eff_2036", "Efficiency 2036 onwards (LHV)", 0.005, "%.3f"),
    ("Electrolysers", "p2h_k0", "Initial capacity (GW)", 0.1, "%.2f"),
    ("Electrolysers", "p2h_growth", "Max capacity growth (/yr)", 0.1, "%.2f"),
    ("Electrolysers", "cert_price_2026", "Renewable certificate price 2026 (A$/MWh)", 1.0, "%.1f"),
    ("Electricity price paid by electrolysers", "elec_2026", "2026 (A$/MWh)", 1.0, "%.0f"),
    ("Electricity price paid by electrolysers", "elec_2030", "2030 (A$/MWh)", 1.0, "%.0f"),
    ("Electricity price paid by electrolysers", "elec_2035", "2035 (A$/MWh)", 1.0, "%.0f"),
    ("Electricity price paid by electrolysers", "elec_2040", "2040 (A$/MWh)", 1.0, "%.0f"),
    ("Electricity price paid by electrolysers", "elec_2045", "2045 (A$/MWh)", 1.0, "%.0f"),
    ("Electricity price paid by electrolysers", "elec_2050", "2050 (A$/MWh)", 1.0, "%.0f"),
    ("Policy and the common fiscal rule", "envelope_m", "Envelope (A$M, FY2027-28 to FY2039-40)", 100.0, "%.0f"),
    ("Policy and the common fiscal rule", "tax_rate", "Tax offset rate (A$/kg)", 0.1, "%.2f"),
    ("Policy and the common fiscal rule", "headstart_cap", "Revenue support cap (A$/kg)", 0.1, "%.2f"),
    ("Policy and the common fiscal rule", "inv_grant_share", "Investment grant share of capex", 0.01, "%.2f"),
    ("Policy and the common fiscal rule", "capex_grant_per_kw", "Capex grant (A$/kW)", 5.0, "%.1f"),
]

# =====================================================================================================================
# SIDEBAR
# =====================================================================================================================
st.sidebar.title("Inputs")
pol_label = st.sidebar.selectbox("Support instrument", [v for k, v in POLICIES.items() if k != "none"], index=2)
policy = [k for k, v in POLICIES.items() if v == pol_label][0]
if st.sidebar.button("Reset all inputs to the paper's calibration"):
    for f in fields(Calibration):
        st.session_state.pop("in_" + f.name, None)
    st.rerun()

with st.sidebar.form("inputs"):
    st.caption("Inputs default to the paper's calibration. Change any value and press Solve.")
    vals, groups = {}, {}
    for g, name, label, step, fmt in INPUTS:
        groups.setdefault(g, []).append((name, label, step, fmt))
    for g, items in groups.items():
        with st.expander(g, expanded=False):
            for name, label, step, fmt in items:
                vals[name] = st.number_input(label, value=float(getattr(DEF, name)), step=step, format=fmt, key="in_" + name)
    vals["safeguard_on"] = st.checkbox("Safeguard Mechanism on SMR", value=True, key="in_safeguard_on")
    st.form_submit_button("Solve", type="primary")

cal = Calibration(**{**asdict(DEF), **vals})


@st.cache_data(show_spinner=False)
def solve(cal_dict, policy):
    out, _, info, _ = run_case(Calibration(**cal_dict), MODEL, policy)
    return out, info


# =====================================================================================================================
# MAIN
# =====================================================================================================================
st.title("Subsidising Green Hydrogen: the domestic hydrogen market")
st.markdown(INTRO)

with st.spinner("Solving the baseline and the policy case..."):
    base, _ = solve(asdict(cal), "none")
    pol, info = solve(asdict(cal), policy)
P = paths(cal)
yrs = base["year"].values
rep = yrs <= 2050
w = base["window"].values

if policy in PER_KG and not info.get("converged", True):
    st.warning(f"The fiscal-rule fixed point stopped after {info['iterations']} solves with a remaining rate gap of "
               f"{info.get('rate_gap', float('nan')):.3f}; results are approximate.")

s = summary(base, pol, cal)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline green H2", f"{base['P2H_kt'][w].sum() / 1000:.2f} Mt", help="Electrolytic hydrogen produced in the policy window with no policy")
c2.metric("Additional with policy", f"{s['additional_kt'] / 1000:.2f} Mt")
c3.metric("Payments (realised)", f"A${s['paid_m'] / 1000:.2f}bn", delta=f"unspent A${s['unspent_m'] / 1000:.2f}bn", delta_color="off")
c4.metric("Cost per additional kg", "n/a" if np.isnan(s["aud_per_additional_kg"]) else f"A${s['aud_per_additional_kg']:.2f}")
st.caption("Window = FY2027-28 to FY2039-40 (shaded). Additional = policy case minus baseline. Payments are real 2025-26 A$, undiscounted.")

tab1, tab2, tab3 = st.tabs(["Production", "Costs and prices", "Fiscal"])
COL = {"SMR": "#8c96a3", "dom": "#3aa776"}


def shade(f):
    f.add_vrect(x0=2027, x1=2039, fillcolor="#f6e7b8", opacity=0.3, line_width=0)
    return f


def mix_fig(df, title):
    f = go.Figure()
    f.add_trace(go.Scatter(x=yrs[rep], y=df["SMR_kt"][rep], name="SMR", stackgroup="a", line=dict(color=COL["SMR"])))
    f.add_trace(go.Scatter(x=yrs[rep], y=df["P2H_dom_kt"][rep], name="Electrolysis", stackgroup="a", line=dict(color=COL["dom"])))
    f.update_layout(title=title, yaxis_title="kt per year", height=420, hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    return shade(f)


with tab1:
    a, b = st.columns(2)
    a.plotly_chart(mix_fig(base, "Baseline (no hydrogen policy)"), width="stretch")
    b.plotly_chart(mix_fig(pol, pol_label), width="stretch")
    f = go.Figure()
    f.add_trace(go.Scatter(x=yrs[rep], y=base["P2H_kt"][rep], name="Baseline", line=dict(color="black", width=3)))
    f.add_trace(go.Scatter(x=yrs[rep], y=pol["P2H_kt"][rep], name=pol_label, line=dict(color="#8e5bd0")))
    f.update_layout(title="Electrolytic hydrogen production", yaxis_title="kt per year", height=380, hovermode="x unified")
    a.plotly_chart(shade(f), width="stretch")
    f = go.Figure()
    f.add_trace(go.Scatter(x=yrs[rep], y=base["K_P2H_GW"][rep], name="Baseline", line=dict(color="black", width=3)))
    f.add_trace(go.Scatter(x=yrs[rep], y=pol["K_P2H_GW"][rep], name=pol_label, line=dict(color="#8e5bd0")))
    f.update_layout(title="Electrolyser capacity", yaxis_title="GW", height=380, hovermode="x unified")
    b.plotly_chart(shade(f), width="stretch")

with tab2:
    f = go.Figure()
    f.add_trace(go.Scatter(x=yrs[rep], y=P["smr_running"][rep], name="SMR running cost incl. Safeguard", line=dict(color="#8c96a3", width=3)))
    f.add_trace(go.Scatter(x=yrs[rep], y=P["p2h_static"][rep], name="Electrolysis, full cost", line=dict(color="#3aa776")))
    f.add_trace(go.Scatter(x=yrs[rep], y=base["dom_H2_price"][rep], name="Domestic hydrogen price (solved), baseline", line=dict(color="black", dash="dot")))
    f.update_layout(title="Costs and prices", yaxis_title="A$/kg", height=440, hovermode="x unified")
    st.plotly_chart(shade(f), width="stretch")
    st.caption("An existing SMR plant keeps running while the domestic price covers its running cost plus the Safeguard "
               "liability, because its capital is sunk.")
    f = go.Figure()
    f.add_trace(go.Scatter(x=yrs[rep], y=base["elec_price_p2h"][rep], name="Electricity price paid by electrolysers (input)", line=dict(color="#3aa776")))
    f.update_layout(title="Electricity price (set in the sidebar)", yaxis_title="A$/MWh", height=340, hovermode="x unified")
    st.plotly_chart(f, width="stretch")

with tab3:
    if policy in PER_KG or policy in GRANTS:
        f = go.Figure()
        f.add_trace(go.Bar(x=yrs[w], y=pol["paid_m"][w], name="Payments (A$M)", marker_color="#8e5bd0"))
        if policy in PER_KG:
            f.add_trace(go.Scatter(x=yrs[w], y=pol["money_available_m"][w], name="Money available", line=dict(color="black", dash="dash")))
        f.update_layout(title="Payments under the common fiscal rule", yaxis_title="A$M (real 2025-26)", height=400)
        st.plotly_chart(f, width="stretch")
        if policy in PER_KG:
            f = go.Figure()
            f.add_trace(go.Scatter(x=yrs[w], y=pol["rate_paid"][w], name="Rate paid (A$/kg)", line=dict(color="#8e5bd0", width=3)))
            f.update_layout(title="Per-kg rate actually paid", yaxis_title="A$/kg", height=320)
            st.plotly_chart(f, width="stretch")
            st.caption(f"Fixed point: {info['iterations']} solves; converged = {info['converged']}."
                       + (f" Contracted volume {info['contract_volume_kt']:.0f} kt/yr." if "contract_volume_kt" in info else ""))
    keep = ["paid_m", "unspent_m", "additional_kt", "aud_per_additional_kg"]
    st.dataframe(pd.DataFrame([{k: round(s[k], 2) for k in keep if k in s}]), hide_index=True)

with st.expander("Year-by-year results"):
    show = pol.drop(columns=["p2h_hourly_GW"], errors="ignore")
    st.dataframe(show[show["year"] <= 2050].round(2), hide_index=True, width="stretch")
    st.download_button("Download policy case (CSV)", show.to_csv(index=False), file_name=f"domestic_{policy}.csv")
    st.download_button("Download baseline (CSV)", base.drop(columns=["p2h_hourly_GW"], errors="ignore").to_csv(index=False),
                       file_name="domestic_baseline.csv")

st.caption("Caleb Hudson, Qazi Haque and Firmin Doko Tchatoka, School of Economics, University of Adelaide. "
           "A simplified version of the paper's model: domestic market only, electricity price given, no exports.")
