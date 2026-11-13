"""
h2_model.py -- simplified, live-solving version of the paper's model
"Subsidising Green Hydrogen: A Multi-Market Equilibrium Model for Australia" (Hudson, Haque, Doko Tchatoka).

Three nested models, each a linear programme solved with HiGHS (scipy.optimize.linprog):
  1. "domestic"  Australian domestic hydrogen market: SMR (with the Safeguard Mechanism) and grid electrolysers meet
                 inelastic domestic demand; the electricity price is an INPUT.
  2. "electric"  as 1, plus the National Electricity Market: solar, wind, hydro, coal, CCGT, OCGT, rooftop and pumped
                 hydro meet grid demand and the electrolysers' load hour by hour; the electricity price is SOLVED.
  3. "full"      as 2, plus exports: Australian hydrogen sells abroad at the delivered price of competing exporters,
                 net of the ammonia export chain and a rising export scale cost.

For given policy rates the competitive equilibrium is the solution of one cost-minimisation problem (the paper's
planner problem), so a single LP solve gives the equilibrium; prices are its duals. Per-kilogram subsidies follow the
paper's common fiscal rule (A$6.7bn, A$515M a year, carry-forward, expiry), and their rates are found as a fixed point
(rate = rule applied to the quantities the rate induces). Grants are imposed inside the LP with the same rule.

Simplifications relative to the paper (so numbers differ from the paper's): FOUR representative days of 24 hours per year
(the paper uses 16), the world market reduced to the price Australia can sell at (the delivered price of competing exporters,
raised by their rising supply cost at the traded share the paper's full model solves for; Australia is a price-taker abroad), no corporate tax, the export scale cost as a piecewise-linear curve. Defaults are the paper's calibration
(Capex1850 v2, 11 November 2026).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

FIRST_YEAR = 2026          # model year 1 = FY2026-27 (financial years starting 1 July)
H = 24
D = 4                      # representative days: one per quarter
SOLAR_DAY = np.array([1.22, 0.94, 0.73, 1.11])   # quarterly multipliers on the solar profile (paper)
WIND_DAY = np.array([0.98, 0.80, 1.25, 0.98])    # quarterly multipliers on the wind profile (paper)

SOLAR_SHAPE = np.array([0, 0, 0, 0, 0, 5, 16, 30, 46, 57, 64, 68, 68, 64, 57, 46, 30, 16, 5, 0, 0, 0, 0, 0]) / 68.0
WIND_SHAPE = np.array([46, 48, 50, 52, 52, 50, 48, 44, 40, 36, 33, 31, 30, 31, 33, 36, 40, 46, 53, 57, 56, 53, 50, 48]) / 57.0
PEAK_SHAPE = np.array([0.68, 0.65, 0.62, 0.62, 0.65, 0.72, 1.00, 1.10, 1.12, 0.90, 0.72, 0.60,
                       0.55, 0.52, 0.56, 0.65, 1.00, 1.62, 1.93, 1.99, 1.87, 1.68, 1.37, 0.90])


def crf(r: float, life: int) -> float:
    return r / (1 - (1 + r) ** (-life))


def interp(knots: list[tuple[float, float]], yr: float) -> float:
    xs, ys = zip(*knots)
    return float(np.interp(yr, xs, ys))


# =====================================================================================================================
# CALIBRATION (defaults = the paper's calibration)
# =====================================================================================================================
@dataclass
class Calibration:
    # general
    years_solved: int = 30                 # 2026-2055 (reported to 2050)
    hours_per_block: int = 2               # time resolution within each representative day (1 = hourly; 2 or 3 solve faster)
    discount_rate: float = 0.05
    # domestic hydrogen demand (kt/yr)
    demand_2026: float = 500.0
    demand_2050: float = 3400.0
    # SMR
    gas_price_2026: float = 13.9           # A$/GJ
    gas_price_2027: float = 14.3
    gas_price_2032: float = 13.75          # flat after
    smr_gas_use: float = 0.175             # GJ (HHV) per kg H2
    smr_capex: float = 1223.0              # A$/kW gas input
    smr_fom_pct: float = 0.03              # of capex per year
    smr_cost_scale: float = 1.05           # fitted scale on gas + fixed O&M
    smr_spread: float = 0.60               # A$/kg spread of running cost across plants
    smr_retire: float = 0.05
    smr_growth: float = 0.17
    # Safeguard Mechanism
    safeguard_on: bool = True
    emissions_intensity: float = 9.5       # kgCO2 per kg H2
    accu_2028: float = 35.0                # A$/t to 2028
    accu_2035: float = 84.0
    accu_growth_after_2035: float = 0.05
    # electrolysers
    p2h_capex_2026: float = 1850.0         # A$/kW
    p2h_capex_2035: float = 1255.4
    p2h_capex_decline_to_2040: float = 0.045
    p2h_capex_decline_after_2040: float = 0.021
    hurdle_2026: float = 0.09              # hurdle rate on new electrolysers, falling linearly to...
    hurdle_2035: float = 0.05
    eff_2026: float = 0.6729               # LHV efficiency
    eff_2030: float = 0.69
    eff_2036: float = 0.80
    p2h_fom_pct: float = 0.03              # of capex per year, spread over output at 60% load
    p2h_var_opex: float = 0.10             # A$/kg
    p2h_k0: float = 0.5                    # GW
    p2h_growth: float = 1.0                # capacity can at most double each year
    p2h_retire: float = 0.05
    p2h_avail: float = 0.95
    cert_price_2026: float = 15.0          # A$/MWh renewable certificates, falling...
    cert_decline: float = 0.05
    # electricity price used by the DOMESTIC-ONLY model and by the static cost estimates (A$/MWh)
    elec_2026: float = 59.0
    elec_2030: float = 109.0
    elec_2035: float = 110.0
    elec_2040: float = 103.0
    elec_2045: float = 90.0
    elec_2050: float = 78.0
    # electricity market (NEM)
    grid_twh_2026: float = 205.0
    grid_twh_2050: float = 355.0
    avg_load_2026_gw: float = 25.8
    solar_k0: float = 12.22
    solar_cf: float = 0.29
    solar_capex_2026: float = 1297.0
    solar_capex_decline: float = 0.03
    wind_k0: float = 14.02
    wind_cf: float = 0.317
    wind_capex_2026: float = 2520.0
    wind_capex_decline: float = 0.015
    ren_running_cost: float = 22.0
    ren_growth: float = 0.40
    ren_allowance_gw: float = 1.0
    solar_retire: float = 0.035
    wind_retire: float = 0.05
    rooftop_2026_gw: float = 25.0
    rooftop_2050_gw: float = 87.0
    rooftop_cf: float = 0.15
    hydro_gw: float = 8.03
    hydro_daily_share: float = 0.16
    hydro_running_cost: float = 5.0
    coal_gw: float = 21.26
    coal_avail: float = 0.825
    coal_running_cost: float = 35.5
    ccgt_gw: float = 5.0
    ccgt_running_cost: float = 120.0
    ccgt_retire: float = 0.02
    ocgt_gw: float = 7.79
    ocgt_running_cost: float = 190.0       # mean, shaped by hour
    ocgt_capex: float = 1500.0
    ocgt_retire: float = 0.02
    gas_avail: float = 0.90
    storage_gw: float = 2.2                # Snowy 2.0 from 2029
    storage_from_year: int = 2029
    storage_roundtrip: float = 0.76
    voll: float = 10000.0                  # A$/MWh unserved energy
    go_matching: bool = True               # annual guarantee-of-origin matching for electrolysers
    # exports and the world price
    h2global_eur_t_nh3: float = 1000.0
    aud_per_eur: float = 1.65
    nh3_h2_share: float = 0.178
    exporter_cost_scale: float = 0.58      # fitted
    shipping: float = 0.343                # A$/kg, ammonia shipping
    nh3_synthesis: float = 0.87            # A$/kg at 85% utilisation
    nh3_synthesis_decline: float = 0.01
    cracking: float = 1.20
    cracking_decline: float = 0.02
    terminal_handling: float = 0.10
    early_synthesis_util: float = 0.60     # first-of-a-kind items, fading linearly to zero by 2035
    buffer_storage: float = 0.40
    foak_premium: float = 1.00             # fitted
    other_exporters_curvature: float = 0.5 # k^O: competitors' delivered price rises by this share if they supplied all world demand
    grey_price: float = 2.70               # world price while the importing regions need no imports
    export_slope: float = 0.125            # A$/kg per Mt/yr exported, 2026
    export_slope_decline: float = 0.03
    # policy
    envelope_m: float = 6700.0             # A$M
    window_first: int = 2027               # FY2027-28
    window_last: int = 2039                # FY2039-40
    tax_rate: float = 2.0                  # A$/kg
    headstart_cap: float = 9.19            # A$/kg (first awarded contract)
    inv_grant_share: float = 0.25
    capex_grant_per_kw: float = 462.5


# =====================================================================================================================
# DERIVED PATHS
# =====================================================================================================================
def paths(c: Calibration) -> dict:
    Y = c.years_solved
    yrs = np.arange(FIRST_YEAR, FIRST_YEAR + Y)
    yi = np.arange(1, Y + 1)
    P = {"years": yrs}
    P["disc"] = (1 + c.discount_rate) ** (-(yi - 1.0))
    # demand
    P["demand"] = np.array([c.demand_2026 + (c.demand_2050 - c.demand_2026) * (min(y, 25) - 1) / 24 for y in yi])
    # SMR
    gas = np.array([interp([(2026, c.gas_price_2026), (2027, c.gas_price_2027), (2032, c.gas_price_2032), (2100, c.gas_price_2032)], t) for t in yrs])
    smr_cap_per_ktyr = c.smr_capex / (0.547 * 365.0)                           # A$M per kt/yr of output capacity
    smr_fom = c.smr_fom_pct * smr_cap_per_ktyr / 0.95                           # A$/kg
    lam = np.array([max(0.0, 0.85 - 0.049 * (t - 2026)) if t <= 2029 else max(0.0, 0.66 - 0.03285 * (t - 2030) * 1.0) for t in yrs])
    accu = np.array([c.accu_2028 if t <= 2028 else (c.accu_2028 + (c.accu_2035 - c.accu_2028) * (t - 2028) / 7 if t <= 2035
                     else c.accu_2035 * (1 + c.accu_growth_after_2035) ** (t - 2035)) for t in yrs])
    P["safeguard"] = c.emissions_intensity * (1 - lam) * accu / 1000.0 if c.safeguard_on else np.zeros(Y)
    P["smr_running"] = (c.smr_gas_use * gas + smr_fom) * c.smr_cost_scale + P["safeguard"]
    P["smr_cap_cost"] = smr_cap_per_ktyr
    # electrolysers
    cx = []
    for t in yrs:
        if t <= 2035:
            cx.append(c.p2h_capex_2026 * (c.p2h_capex_2035 / c.p2h_capex_2026) ** ((t - 2026) / 9))
        elif t <= 2040:
            cx.append(c.p2h_capex_2035 * (1 - c.p2h_capex_decline_to_2040) ** (t - 2035))
        else:
            cx.append(c.p2h_capex_2035 * (1 - c.p2h_capex_decline_to_2040) ** 5 * (1 - c.p2h_capex_decline_after_2040) ** (t - 2040))
    P["p2h_capex"] = np.array(cx)
    P["hurdle"] = np.array([c.hurdle_2026 + (c.hurdle_2035 - c.hurdle_2026) * min(1.0, (t - 2026) / 9) for t in yrs])
    P["eff"] = np.array([interp([(2026, c.eff_2026), (2030, c.eff_2030), (2036, c.eff_2036), (2100, c.eff_2036)], t) for t in yrs])
    P["p2h_opex"] = c.p2h_fom_pct * P["p2h_capex"] / (8760 * 0.6 * P["eff"] / 33.33) + c.p2h_var_opex   # A$/kg
    P["cert"] = c.cert_price_2026 * (1 - c.cert_decline) ** (yi - 1)
    P["elec_static"] = np.array([interp([(2026, c.elec_2026), (2030, c.elec_2030), (2035, c.elec_2035), (2040, c.elec_2040),
                                         (2045, c.elec_2045), (2050, c.elec_2050), (2055, 64.0), (2100, 64.0)], t) for t in yrs])
    # export chain (Australia) and the delivered price of competing exporters
    base_chain = c.shipping + c.nh3_synthesis * (1 - c.nh3_synthesis_decline) ** (yi - 1) + c.cracking * (1 - c.cracking_decline) ** (yi - 1) + c.terminal_handling
    fade = np.array([max(0.0, 1 - (y - 1) / 9) for y in yi])                   # 1 in 2026 -> 0 from 2035
    def synth(u, r):
        return (770.0 * crf(r, 20) / u + 0.03 * 770.0 + 0.61 * 60.0 * 0.70) / 177.0 / 0.70
    util = c.early_synthesis_util + (0.85 - c.early_synthesis_util) * (1 - fade)
    extra = np.array([max(0.0, synth(util[i], P["hurdle"][i]) - synth(0.85, 0.05)) for i in range(Y)])
    P["chain"] = base_chain + extra + c.buffer_storage * fade + c.foak_premium * fade
    rev = lambda t: 5.3 + (4.4 - 5.3) * (t - 2020) / 10 if t <= 2030 else (4.4 + (2.7 - 4.4) * (t - 2030) / 20 if t <= 2050 else 2.7)
    lcoh27 = c.h2global_eur_t_nh3 / 1000 / c.nh3_h2_share * c.aud_per_eur - (base_chain[1] - c.cracking * (1 - c.cracking_decline) - c.terminal_handling)
    P["import_price"] = np.array([c.exporter_cost_scale * lcoh27 * rev(t) / rev(2027) for t in yrs]) + base_chain
    # share of world demand supplied by other exporters in the paper's full-model baseline (world market solved there)
    share = np.array([interp([(2026, 0.0), (2028, 0.126), (2030, 0.13), (2032, 0.211), (2034, 0.269), (2036, 0.305), (2038, 0.324),
                              (2040, 0.385), (2042, 0.421), (2044, 0.435), (2046, 0.433), (2048, 0.424), (2050, 0.425), (2100, 0.425)], t) for t in yrs])
    P["other_share"] = share
    P["export_price"] = np.where(share < 0.01, c.grey_price, P["import_price"] * (1 + c.other_exporters_curvature * share))
    P["export_slope"] = c.export_slope * (1 - c.export_slope_decline) ** (np.minimum(yi, 25) - 1)
    # static costs (for the cost-gap rates)
    ann = np.array([crf(P["hurdle"][i], 20) for i in range(Y)])
    P["p2h_static"] = P["p2h_opex"] + ann * P["p2h_capex"] / (0.95 * 8760 * P["eff"] / 33.33) + (P["elec_static"] + P["cert"]) * 33.33 / P["eff"] / 1000
    P["smr_static"] = P["smr_running"] + crf(c.discount_rate, 20) * smr_cap_per_ktyr / 0.95
    P["gap"] = P["p2h_static"] - P["smr_static"]
    # electricity
    twh = np.array([c.grid_twh_2026 + (c.grid_twh_2050 - c.grid_twh_2026) * (min(y, 25) - 1) / 24 for y in yi])
    P["grid_load"] = c.avg_load_2026_gw * twh / c.grid_twh_2026
    blk = int(c.hours_per_block); assert 24 % blk == 0, "hours per block must divide 24"
    P["blk"], P["H"] = blk, 24 // blk
    agg = lambda v: np.asarray(v).reshape(24 // blk, blk).mean(axis=1)
    P["solar_cf"] = agg(SOLAR_SHAPE) * c.solar_cf / SOLAR_SHAPE.mean()
    P["wind_cf"] = agg(WIND_SHAPE) * c.wind_cf / WIND_SHAPE.mean()
    P["roof_cf"] = agg(SOLAR_SHAPE) * c.rooftop_cf / SOLAR_SHAPE.mean()
    P["rooftop"] = np.array([c.rooftop_2026_gw + (c.rooftop_2050_gw - c.rooftop_2026_gw) * (min(y, 25) - 1) / 24 for y in yi])
    coal_share = np.array([interp([(2026, 1.0), (2028, 1.0), (2030, 0.8), (2035, 0.4), (2038, 0.2), (2043, 0.1), (2049, 0.0), (2100, 0.0)], t) for t in yrs])
    P["coal"] = c.coal_gw * coal_share
    P["ccgt"] = c.ccgt_gw * (1 - c.ccgt_retire) ** (yi - 1)
    P["solar_capex"] = c.solar_capex_2026 * (1 - c.solar_capex_decline) ** (yi - 1)
    P["wind_capex"] = c.wind_capex_2026 * (1 - c.wind_capex_decline) ** (yi - 1)
    P["ocgt_cost_h"] = c.ocgt_running_cost * agg(PEAK_SHAPE) / PEAK_SHAPE.mean()
    P["storage"] = np.array([c.storage_gw if t >= c.storage_from_year else 0.0 for t in yrs])
    P["window"] = np.array([c.window_first <= t <= c.window_last for t in yrs])
    return P


def annuity_weight(cost_y: np.ndarray, rates: np.ndarray, life: int, disc: np.ndarray) -> np.ndarray:
    """c~_y: discounted annuity payments of a build in year y, inside the solved horizon, per unit, valued at year y."""
    Y = len(cost_y)
    out = np.zeros(Y)
    for y in range(Y):
        k = crf(rates[y], life)
        out[y] = cost_y[y] * k * sum(disc[t] / disc[y] for t in range(y, min(Y, y + life)))
    return out


# =====================================================================================================================
# LP BUILDER
# =====================================================================================================================
class LP:
    def __init__(self):
        self.n = 0
        self.lb, self.ub, self.c = [], [], []
        self.eq_r, self.eq_c, self.eq_v, self.eq_b = [], [], [], []
        self.ub_r, self.ub_c, self.ub_v, self.ub_b = [], [], [], []
        self.idx = {}

    def var(self, name, shape, lb=0.0, ub=None, cost=0.0):
        size = int(np.prod(shape))
        ids = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        lbv = np.broadcast_to(np.asarray(lb, float), shape).ravel()
        ubv = np.broadcast_to(np.asarray(np.inf if ub is None else ub, float), shape).ravel()
        cv = np.broadcast_to(np.asarray(cost, float), shape).ravel()
        self.lb += list(lbv); self.ub += list(ubv); self.c += list(cv)
        self.idx[name] = ids
        return ids

    def eq(self, terms, rhs):
        r = len(self.eq_b)
        for i, v in terms:
            self.eq_r.append(r); self.eq_c.append(int(i)); self.eq_v.append(float(v))
        self.eq_b.append(float(rhs))
        return r

    def le(self, terms, rhs):
        r = len(self.ub_b)
        for i, v in terms:
            self.ub_r.append(r); self.ub_c.append(int(i)); self.ub_v.append(float(v))
        self.ub_b.append(float(rhs))
        return r

    def solve(self):
        A_eq = coo_matrix((self.eq_v, (self.eq_r, self.eq_c)), shape=(len(self.eq_b), self.n)).tocsr() if self.eq_b else None
        A_ub = coo_matrix((self.ub_v, (self.ub_r, self.ub_c)), shape=(len(self.ub_b), self.n)).tocsr() if self.ub_b else None
        bounds = list(zip(self.lb, [None if np.isinf(u) else u for u in self.ub]))
        res = linprog(np.array(self.c), A_ub=A_ub, b_ub=np.array(self.ub_b) if self.ub_b else None,
                      A_eq=A_eq, b_eq=np.array(self.eq_b) if self.eq_b else None, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP not solved: {res.message}")
        return res


POLICIES = {
    "none": "Baseline (no hydrogen policy)",
    "rs_gap": "Revenue support (cost gap, contracted volume)",
    "rs_fixed": "Revenue support (A$9.19/kg, contracted volume)",
    "tax": "Production tax offset (A$2/kg)",
    "cfd": "Contract for difference",
    "inv_grant": "Investment grant (25% of capex)",
    "capex_grant": "Capex grant (A$462.50/kW)",
}
PER_KG = {"rs_gap", "rs_fixed", "tax", "cfd"}
GRANTS = {"inv_grant", "capex_grant"}


def announced_rate(policy: str, c: Calibration, P: dict) -> np.ndarray:
    w = P["window"]
    if policy == "tax":
        return np.where(w, c.tax_rate, 0.0)
    if policy == "cfd":
        return np.where(w, P["gap"], 0.0)                      # two-way: negative gap = clawback
    if policy == "rs_gap":
        return np.where(w, np.clip(P["gap"], 0, c.headstart_cap), 0.0)
    if policy == "rs_fixed":
        return np.where(w, c.headstart_cap, 0.0)
    return np.zeros(len(w))


# =====================================================================================================================
# ONE LP SOLVE AT GIVEN PER-KG RATES
# =====================================================================================================================
def solve_once(c: Calibration, P: dict, model: str, policy: str, rates: np.ndarray, volume_cap: float | None):
    Y = c.years_solved
    disc = P["disc"]
    lp = LP()
    w = P["window"]
    elec = model in ("electric", "full")
    exports = model == "full"
    blk = P["blk"]; H = P["H"]
    day = 365.0 / D * blk                     # each block lasts blk hours

    # ---- electrolysers ----
    hurdle = P["hurdle"]
    p2h_ann = annuity_weight(P["p2h_capex"], hurdle, 20, disc)                 # A$M per GW built
    K = lp.var("K_p2h", (Y,))
    b = lp.var("b_p2h", (Y,), cost=disc * p2h_ann)
    e = lp.var("e", (Y, D, H))                                                     # GW in each hour
    for y in range(Y):
        lp.eq([(K[y], 1.0), (b[y], -1.0)] + ([(K[y - 1], -(1 - c.p2h_retire))] if y else []), (1 - c.p2h_retire) * c.p2h_k0 if y == 0 else 0.0)
        lp.le([(b[y], 1.0)] + ([(K[y - 1], -c.p2h_growth)] if y else []), c.p2h_growth * c.p2h_k0 if y == 0 else 0.0)
        for d in range(D):
            for h in range(H):
                lp.le([(e[y, d, h], 1.0), (K[y], -c.p2h_avail)], 0.0)
    kt_per_gwh = P["eff"] / 33.33                                               # kt H2 per GWh
    # hydrogen output g_y (kt/yr) = 365 * sum_h e * eff/33.33 ; split into domestic and exported
    gdom = lp.var("g_dom", (Y,))
    # running cost of electrolysers (non-electricity) + certificates; electricity: input price (domestic) or system cost
    # certificates are a cost to electrolysers but income to renewables, so they cancel when the NEM is in the model
    for y in range(Y):
        for d in range(D):
            for h in range(H):
                lp.c[e[y, d, h]] += disc[y] * day * (P["p2h_opex"][y] * kt_per_gwh[y]
                                                     + (0.0 if elec else (P["elec_static"][y] + P["cert"][y]) / 1000.0))
    # subsidies on output
    if policy in PER_KG:
        sup = lp.var("g_sup", (Y,))
        for y in range(Y):
            lp.c[sup[y]] += -disc[y] * rates[y]
            lp.le([(sup[y], 1.0)] + [(e[y, d, h], -day * kt_per_gwh[y]) for d in range(D) for h in range(H)], 0.0)
            if volume_cap is not None:
                lp.ub[sup[y]] = volume_cap if w[y] else 0.0
            elif not w[y]:
                lp.ub[sup[y]] = 0.0
            if policy in ("tax", "cfd") and w[y]:                               # paid on ALL output in the window: supported = total
                lp.eq([(sup[y], 1.0)] + [(e[y, d, h], -day * kt_per_gwh[y]) for d in range(D) for h in range(H)], 0.0)
    if policy in GRANTS:
        bG = lp.var("b_grant", (Y,))
        if policy == "inv_grant":
            gc = c.inv_grant_share * P["p2h_capex"]
        else:
            gc = np.full(Y, c.capex_grant_per_kw)
        grant_credit = gc / P["p2h_capex"] * p2h_ann
        a = c.envelope_m / w.sum()
        cum_alloc = 0.0
        prior = []
        for y in range(Y):
            lp.c[bG[y]] += -disc[y] * grant_credit[y]
            lp.le([(bG[y], 1.0), (b[y], -1.0)], 0.0)
            if not w[y]:
                lp.ub[bG[y]] = 0.0
                continue
            cum_alloc = min(cum_alloc + a, c.envelope_m)
            prior.append((bG[y], gc[y]))
            lp.le(list(prior), cum_alloc)                                      # cumulative payments <= cumulative allocation
        P_grant_cost = gc
    # ---- SMR and the domestic market ----
    smr_ann = annuity_weight(np.full(Y, P["smr_cap_cost"]), np.full(Y, c.discount_rate), 20, disc)
    Ks = lp.var("K_smr", (Y,))
    bs = lp.var("b_smr", (Y,), cost=disc * smr_ann)
    tr = lp.var("smr_tranche", (Y, 4))
    k0s = P["demand"][0] / 0.95
    offs = np.array([-0.375, -0.125, 0.125, 0.375]) * c.smr_spread
    dom_rows = []
    for y in range(Y):
        lp.eq([(Ks[y], 1.0), (bs[y], -1.0)] + ([(Ks[y - 1], -(1 - c.smr_retire))] if y else []), (1 - c.smr_retire) * k0s if y == 0 else 0.0)
        lp.le([(bs[y], 1.0)] + ([(Ks[y - 1], -c.smr_growth)] if y else []), (c.smr_growth + c.smr_retire) * k0s if y == 0 else 0.0)
        lp.le([(tr[y, k], 1.0) for k in range(4)] + [(Ks[y], -0.95)], 0.0)
        for k in range(4):
            lp.ub[tr[y, k]] = P["demand"][y] / 4
            lp.c[tr[y, k]] += disc[y] * (P["smr_running"][y] + offs[k])
        dom_rows.append(lp.eq([(tr[y, k], 1.0) for k in range(4)] + [(gdom[y], 1.0)], P["demand"][y]))
    # ---- exports ----
    seg_w, nseg = 500.0, 60
    if exports:
        xs = lp.var("x_seg", (Y, nseg), ub=seg_w)
        for y in range(Y):
            for k in range(nseg):
                lp.c[xs[y, k]] += disc[y] * (P["chain"][y] + P["export_slope"][y] * (k + 0.5) * seg_w / 1000.0 - P["export_price"][y])
            lp.eq([(gdom[y], 1.0)] + [(xs[y, k], 1.0) for k in range(nseg)] + [(e[y, d, h], -day * kt_per_gwh[y]) for d in range(D) for h in range(H)], 0.0)
    else:
        for y in range(Y):
            lp.eq([(gdom[y], 1.0)] + [(e[y, d, h], -day * kt_per_gwh[y]) for d in range(D) for h in range(H)], 0.0)
    # ---- electricity market ----
    bal_rows = None
    if elec:
        sol_ann = annuity_weight(P["solar_capex"], np.full(Y, c.discount_rate), 30, disc)
        wnd_ann = annuity_weight(P["wind_capex"], np.full(Y, c.discount_rate), 30, disc)
        gas_ann = annuity_weight(np.full(Y, c.ocgt_capex), np.full(Y, c.discount_rate), 30, disc)
        caps = {}
        for name, k0, ret, ann, grow, allow in [("solar", c.solar_k0, c.solar_retire, sol_ann, c.ren_growth, c.ren_allowance_gw),
                                                ("wind", c.wind_k0, c.wind_retire, wnd_ann, c.ren_growth, c.ren_allowance_gw),
                                                ("ocgt", c.ocgt_gw, c.ocgt_retire, gas_ann, 1.0, 5.0)]:
            Kx = lp.var("K_" + name, (Y,)); bx = lp.var("b_" + name, (Y,), cost=disc * ann)
            for y in range(Y):
                lp.eq([(Kx[y], 1.0), (bx[y], -1.0)] + ([(Kx[y - 1], -(1 - ret))] if y else []), (1 - ret) * k0 if y == 0 else 0.0)
                lp.le([(bx[y], 1.0)] + ([(Kx[y - 1], -grow)] if y else []), grow * k0 + allow if y == 0 else allow)
            caps[name] = Kx
        g = {}
        for name in ["solar", "wind", "ocgt", "coal", "ccgt", "hydro", "roof", "unserved"]:
            g[name] = lp.var("g_" + name, (Y, D, H))
        ch = lp.var("ch", (Y, D, H)); dis = lp.var("dis", (Y, D, H)); soc = lp.var("soc", (Y, D, H))
        eta = math.sqrt(c.storage_roundtrip)
        bal_rows = np.zeros((Y, D, H), dtype=int)
        for y in range(Y):
            for d in range(D):
                for h in range(H):
                    lp.le([(g["solar"][y, d, h], 1.0), (caps["solar"][y], -min(1.0, P["solar_cf"][h] * SOLAR_DAY[d]))], 0.0)
                    lp.le([(g["wind"][y, d, h], 1.0), (caps["wind"][y], -min(1.0, P["wind_cf"][h] * WIND_DAY[d]))], 0.0)
                    lp.le([(g["ocgt"][y, d, h], 1.0), (caps["ocgt"][y], -c.gas_avail)], 0.0)
                    lp.ub[g["coal"][y, d, h]] = P["coal"][y] * c.coal_avail
                    lp.ub[g["ccgt"][y, d, h]] = P["ccgt"][y] * c.gas_avail
                    lp.ub[g["hydro"][y, d, h]] = c.hydro_gw * 0.9
                    lp.ub[g["roof"][y, d, h]] = P["rooftop"][y] * min(1.0, P["roof_cf"][h] * SOLAR_DAY[d])
                    lp.ub[ch[y, d, h]] = P["storage"][y]; lp.ub[dis[y, d, h]] = P["storage"][y]
                    lp.ub[soc[y, d, h]] = P["storage"][y] * 24 / blk
                    for name, oc in [("solar", c.ren_running_cost), ("wind", c.ren_running_cost), ("coal", c.coal_running_cost),
                                     ("ccgt", c.ccgt_running_cost), ("hydro", c.hydro_running_cost), ("roof", 0.0),
                                     ("ocgt", P["ocgt_cost_h"][h]), ("unserved", c.voll)]:
                        lp.c[g[name][y, d, h]] += disc[y] * day * oc / 1000.0
                    prev = soc[y, d, h - 1] if h else soc[y, d, H - 1]
                    lp.eq([(soc[y, d, h], 1.0), (prev, -1.0), (ch[y, d, h], -eta), (dis[y, d, h], 1.0 / eta)], 0.0)
                    bal_rows[y, d, h] = lp.eq([(g[n][y, d, h], 1.0) for n in g] + [(dis[y, d, h], 1.0), (ch[y, d, h], -1.0), (e[y, d, h], -1.0)],
                                              P["grid_load"][y])
                lp.le([(g["hydro"][y, d, h], blk) for h in range(H)], c.hydro_daily_share * 24 * c.hydro_gw)
            if c.go_matching:
                lp.le([(e[y, d, h], 1.0) for d in range(D) for h in range(H)]
                      + [(g[n][y, d, h], -1.0) for n in ("solar", "wind", "hydro") for d in range(D) for h in range(H)], 0.0)
    res = lp.solve()
    x = res.x
    out = pd.DataFrame({"year": P["years"]})
    gtot = np.array([day * kt_per_gwh[y] * x[e[y]].sum() for y in range(Y)])
    out["P2H_kt"] = gtot
    out["P2H_dom_kt"] = x[gdom]
    out["exports_kt"] = gtot - x[gdom]
    out["SMR_kt"] = x[tr].sum(axis=1)
    out["K_P2H_GW"] = x[K]
    out["K_SMR_ktyr"] = x[Ks]
    out["P2H_build_GW"] = x[b]
    eq_m = res.eqlin.marginals
    out["dom_H2_price"] = np.array([eq_m[dom_rows[y]] / disc[y] for y in range(Y)])
    if elec:
        price = np.array([[[eq_m[bal_rows[y, d, h]] * 1000.0 / (disc[y] * day) for h in range(H)] for d in range(D)] for y in range(Y)])
        out["elec_price"] = price.mean(axis=(1, 2))
        ex = x[e]
        out["elec_price_p2h"] = np.where(ex.sum(axis=(1, 2)) > 1e-6, (price * ex).sum(axis=(1, 2)) / np.maximum(ex.sum(axis=(1, 2)), 1e-9), np.nan)
        price_h = price.mean(axis=1)
        for n in ("solar", "wind"):
            out[f"K_{n}_GW"] = x[lp.idx["K_" + n]]
        out["unserved_GWh"] = day * x[lp.idx["g_unserved"]].sum(axis=(1, 2))
        hourly = price_h
    else:
        out["elec_price"] = P["elec_static"]
        out["elec_price_p2h"] = P["elec_static"]
        hourly = np.tile(P["elec_static"][:, None], (1, H))
    out["p2h_hourly_GW"] = list(x[e].mean(axis=1))
    if policy in PER_KG:
        out["eligible_kt"] = x[lp.idx["g_sup"]]
    elif policy in GRANTS:
        out["granted_GW"] = x[lp.idx["b_grant"]]
        out["grant_paid_m"] = x[lp.idx["b_grant"]] * P_grant_cost
    return out, hourly


# =====================================================================================================================
# POLICY CASE: fixed point on the per-kg rate (the common fiscal rule)
# =====================================================================================================================
def fiscal_rule(c: Calibration, P: dict, announced: np.ndarray, eligible: np.ndarray, whole_envelope: bool = False):
    """Rate paid each year = min(announced, money available / eligible output); unspent money carries forward.
    whole_envelope=True (the tax offset): the whole envelope is available from the first year, so the rate is paid in
    full until the money runs out."""
    w = P["window"]
    a = c.envelope_m / w.sum()
    paid_before, alloc = 0.0, 0.0
    rate = np.zeros_like(announced, dtype=float)
    avail = np.zeros_like(rate)
    for y in range(len(rate)):
        if not w[y]:
            continue
        alloc = c.envelope_m if whole_envelope else min(alloc + a, c.envelope_m)
        M = max(alloc - paid_before, 0.0)
        avail[y] = M
        q = max(eligible[y], 1e-9)
        r = announced[y]
        if r > 0:
            r = min(r, M / q)
        rate[y] = r
        paid_before += r * eligible[y]
    return rate, avail


def run_case(c: Calibration, model: str, policy: str, max_iter: int = 30, damping: float = 0.5, tol: float = 3e-2):
    """Solve one case. Per-kg instruments: fixed point on the rates, rate = fiscal rule(quantities at that rate),
    with damping that halves whenever the rates stop getting closer (the LP can jump between corners)."""
    P = paths(c)
    Y = c.years_solved
    info = {"iterations": 1, "converged": True}
    if policy in PER_KG:
        ann = announced_rate(policy, c, P)
        vol = None
        if policy in ("rs_gap", "rs_fixed"):
            vol = c.envelope_m / max(ann[P["window"]].sum(), 1e-9)              # contracted volume, kt/yr
            info["contract_volume_kt"] = vol
        rate = ann.copy()
        info["converged"] = False
        for it in range(max_iter):
            out, hourly = solve_once(c, P, model, policy, rate, vol)
            new_rate, avail = fiscal_rule(c, P, ann, out["eligible_kt"].values, policy == "tax")
            gap = np.max(np.abs(new_rate - rate)) / max(np.max(np.abs(ann)), 1e-9)
            info["iterations"] = it + 1
            if gap < tol:
                info["converged"] = True
                break
            lam_k = max(damping / (1 + it / 4), 0.02)                          # decreasing steps (averaging) damp the cycling
            rate = rate + lam_k * (new_rate - rate)
        info["rate_gap"] = float(gap)
        rule_rate, avail = fiscal_rule(c, P, ann, out["eligible_kt"].values, policy == "tax")
        out["rate_paid"] = rule_rate                                            # the government pays the rule at the solved quantities
        out["rate_firms"] = rate                                                # the rate firms acted on
        out["money_available_m"] = avail
        out["paid_m"] = rule_rate * out["eligible_kt"].values
    elif policy in GRANTS:
        out, hourly = solve_once(c, P, model, policy, np.zeros(Y), None)
        out["paid_m"] = out["grant_paid_m"]
        out["money_available_m"] = np.nan
    else:
        out, hourly = solve_once(c, P, model, policy, np.zeros(Y), None)
        out["paid_m"] = 0.0
    out["window"] = P["window"]
    return out, hourly, info, P


def summary(base: pd.DataFrame, pol: pd.DataFrame, c: Calibration) -> dict:
    w = base["window"].values
    paid = pol["paid_m"].fillna(0).values[w].sum()
    add = (pol["P2H_kt"].values[w] - base["P2H_kt"].values[w]).sum()
    add_x = (pol["exports_kt"].values[w] - base["exports_kt"].values[w]).sum()
    add_d = (pol["P2H_dom_kt"].values[w] - base["P2H_dom_kt"].values[w]).sum()
    disc = (1 + c.discount_rate) ** (-(np.arange(len(w))))
    return {"allocated_m": c.envelope_m, "paid_m": paid, "paid_pv_m": float((pol["paid_m"].fillna(0).values * disc)[w].sum()),
            "unspent_m": c.envelope_m - paid, "additional_kt": add, "additional_exports_kt": add_x,
            "additional_domestic_kt": add_d, "aud_per_additional_kg": paid / add if add > 1e-6 else float("nan")}


if __name__ == "__main__":
    import time
    c = Calibration()
    for m in ("domestic", "electric", "full"):
        t0 = time.time()
        b, _, _, _ = run_case(c, m, "none")
        print(m, f"{time.time()-t0:.1f}s", "P2H FY2039-40", round(b.loc[b.year == 2039, "P2H_kt"].item()), "exports", round(b.loc[b.year == 2039, "exports_kt"].item()),
              "P2H 2050", round(b.loc[b.year == 2050, "P2H_kt"].item()), "price 2039", round(b.loc[b.year == 2039, "elec_price"].item(), 1))
