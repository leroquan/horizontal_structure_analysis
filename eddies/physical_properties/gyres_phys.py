"""Relations physiques pour classifier les systèmes en rotation du Léman.

Fonctions pures (numpy seulement), utilisées par `regimes_interactifs.ipynb`.
Les seuils de régime sont des ordres de grandeur (voir notes/physique_et_diagnostics.md)
et sont à calibrer sur le modèle.
"""
from __future__ import annotations

import numpy as np

OMEGA = 7.292e-5  # vitesse de rotation de la Terre (rad/s)
G = 9.81  # m/s²


# --------------------------------------------------------------------------- base
def coriolis(lat_deg):
    return 2 * OMEGA * np.sin(np.deg2rad(lat_deg))


def inertial_period_h(f):
    return 2 * np.pi / f / 3600.0


def rho_water(T):
    """Masse volumique de l'eau pure (UNESCO, 1 atm), T en °C."""
    T = np.asarray(T, dtype=float)
    return (
        999.842594
        + 6.793952e-2 * T
        - 9.095290e-3 * T**2
        + 1.001685e-4 * T**3
        - 1.120083e-6 * T**4
        + 6.536332e-9 * T**5
    )


def thermal_expansion(T):
    """Coefficient de dilatation thermique α = -(1/ρ) dρ/dT (1/K)."""
    return float(-(rho_water(T + 0.5) - rho_water(T - 0.5)) / rho_water(T))


def reduced_gravity(T_epi, T_hypo):
    """g' = g (ρ_hypo - ρ_epi)/ρ_hypo ; négatif si la colonne est instable."""
    r1, r2 = rho_water(T_epi), rho_water(T_hypo)
    return float(G * (r2 - r1) / r2)


def internal_wave_speed(gp, h1, H):
    """Vitesse du mode interne 1 d'un modèle à deux couches."""
    h2 = max(H - h1, 1e-6)
    return float(np.sqrt(max(gp, 0.0) * h1 * h2 / (h1 + h2)))


# Résolution effective du modèle, en mailles de DIAMÈTRE (rayon minimal résolu = N_EFF/2 mailles).
# Règle empirique (5–8 mailles dans la littérature, à vérifier avec le spectre d'énergie du modèle).
N_EFF_DEFAULT = 8.0


def lake(lat, T_epi, T_hypo, h1, H, N_th, N_hypo, dx=100.0, dz=1.0, n_eff=N_EFF_DEFAULT, **_):
    """Paramètres de base du lac (dictionnaire)."""
    f = float(coriolis(lat))
    gp = reduced_gravity(T_epi, T_hypo)
    c = internal_wave_speed(max(gp, 1e-5), h1, H)  # plancher si colonne instable
    return dict(
        f=f,
        Ti=inertial_period_h(f),
        gp=gp,
        c=c,
        Ri=c / f,
        N_th=N_th,
        N_hypo=N_hypo,
        H=H,
        h1=h1,
        dx=dx,
        dz=dz,
        n_eff=n_eff,
        rho_epi=float(rho_water(T_epi)),
        rho_hypo=float(rho_water(T_hypo)),
        T_epi=T_epi,
        T_hypo=T_hypo,
    )


# ------------------------------------------------------------ nombres sans dimension
def rossby(U, L, f):
    return U / (f * L)


def burger(Ri, L):
    return (Ri / L) ** 2


def vertical_scale(f, L, N):
    """Échelle verticale quasi-géostrophique H = f L / N."""
    return f * L / N


def ray_slope(omega_over_f, N_over_f):
    """Pente du rayon d'onde interne : tanθ = √((ω²−f²)/(N²−ω²)), NaN hors de (f, N)."""
    w = np.atleast_1d(np.asarray(omega_over_f, dtype=float))
    out = np.full(w.shape, np.nan)
    ok = (w > 1) & (w < N_over_f)
    out[ok] = np.sqrt((w[ok] ** 2 - 1) / (N_over_f**2 - w[ok] ** 2))
    return out


# ---------------------------------------------------------------- Eady (front)
def eady_growth(k, N, f, H, Lam):
    """Taux de croissance d'Eady σ(k) (1/s) ; Lam = dU/dz."""
    mu = N * np.asarray(k, dtype=float) * H / f
    half = mu / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (1.0 / np.tanh(half) - half) * (half - np.tanh(half))
    arg = np.where(np.isfinite(arg) & (arg > 0), arg, 0.0)
    return f * Lam / N * np.sqrt(arg)


def eady_lambda_max(Rd):
    return 3.93 * Rd


def eady_sigma_max(f, Lam, N):
    return 0.3098 * f * Lam / N


# --------------------------------------------------------------------------- vent
def drag_coefficient(U10):
    """Coefficient de traînée (Large & Pond simplifié)."""
    return (1.2e-3 if U10 < 11 else (0.49 + 0.065 * U10) * 1e-3)


def wind_stress(U10, rho_air=1.2):
    return rho_air * drag_coefficient(U10) * U10**2


def wedderburn(gp, h1, tau, L, rho=1000.0):
    ustar2 = tau / rho
    return gp * h1**2 / (ustar2 * L)


# ----------------------------------------------------------------- vent de gradient
def gaussian_vortex(r, V0, R):
    """Tourbillon V(r)=V0 x exp((1-x²)/2), x=r/R ; |V|max = |V0| en r = R."""
    x = np.asarray(r, dtype=float) / R
    V = V0 * x * np.exp((1 - x**2) / 2)
    zeta = V0 / R * (2 - x**2) * np.exp((1 - x**2) / 2)
    return V, zeta


# ------------------------------------------------------------------ classification
def ro_regime(Ro):
    a = abs(Ro)
    if a < 0.1:
        return "géostrophique (abs(Ro) < 0,1)"
    if a < 0.5:
        return "cyclo-géostrophique (0,1 ≤ abs(Ro) < 0,5)"
    if a < 1:
        return "fortement agéostrophique (0,5 ≤ abs(Ro) < 1)"
    return "agéostrophique, rotation secondaire (abs(Ro) ≥ 1)"


def ke_regime(Ke):
    if Ke < 0.3:
        return "quasi non tournant (L ≪ Rᵢ)"
    if Ke < 1:
        return "transition (L < Rᵢ)"
    if Ke <= 3:
        return "rotationnel (L ~ Rᵢ : Kelvin, Poincaré, gyres baroclines)"
    return "rotation dominante (L ≫ Rᵢ : colonnaire / barotrope)"


def wave_band(w_over_f, N_over_f):
    if w_over_f < 0.8:
        return "sous-inertiel (ω < f) : Kelvin, topographique, réponse forcée non propagative"
    if w_over_f <= 1.2:
        return "quasi-inertiel (0,8–1,2 f) : oscillations inertielles, propagation presque horizontale"
    if w_over_f < N_over_f:
        return "super-inertiel (f < ω < N) : Poincaré / ondes internes"
    return "ω > N : évanescent (ondes de surface, turbulence)"


def slope_regime(s, sc):
    if not np.isfinite(sc):
        return "pas d'onde libre à cette fréquence"
    r = s / sc
    if r < 0.8:
        return f"sous-critique (s/s_c = {r:.2f}) : réflexion vers l'avant"
    if r <= 1.25:
        return f"critique (s/s_c = {r:.2f}) : amplification, déferlement, mélange"
    return f"sur-critique (s/s_c = {r:.2f}) : réflexion vers le large"


def grid_slope_regime(s, sg):
    r = s / sg
    if r < 0.5:
        return "sous-résolu : marches larges (terrasses)"
    if r <= 2:
        return "escalier ≈ 1 marche par maille"
    return "mur : plusieurs niveaux par maille"


def eps_regime(eps):
    if eps < 0.1:
        return "négligeable (couches quasi fixes)"
    if eps < 0.5:
        return "modéré (couche du haut ±10–50 %)"
    if eps < 1:
        return "élevé (couche du haut presque annulée)"
    return "critique : la couche s'annule → z* ou couche épaisse obligatoire"


def cap_regime(Ro_cap, S):
    if S >= 1:
        return "sillage attaché, stationnaire (friction forte, S ≳ 1)"
    if Ro_cap < 1:
        return "la rotation domine : suit les isobathes, séparation peu probable (Ro_cap ≲ 1)"
    return "séparation instationnaire probable : émission de tourbillons (Ro_cap ≳ 1, S ≪ 1)"


def plume_regime(rho_r, rho_s, rho_h):
    if rho_r < rho_s:
        return "plus léger que la surface → panache de surface (bulbe, courant côtier)"
    if rho_r <= rho_h:
        return "entre surface et fond → intrusion (interflow) dans la thermocline"
    return "plus lourd que l'hypolimnion → plongée jusqu'au fond (courant de gravité / turbidité)"


def plume_kelvin_regime(Ke):
    if Ke < 0.3:
        return "étalement quasi radial (rotation négligeable près de l'embouchure)"
    if Ke < 1:
        return "transition"
    return "virage à droite : bulbe anticyclonique puis courant côtier (sens de Kelvin)"


# ------------------------------------------------------------------ onde de Kelvin
def kelvin_zeta_over_f(u0, c, f, kappa, d):
    """ζ/f d'une onde de Kelvin interne (côte à droite) à la distance d de la côte, autour d'un cap.

    Courant le long de la côte u = u0 exp(-d/Rᵢ), Rᵢ = c/f. Courbure de la côte κ (> 0 : cap, centre de
    courbure côté terre ; < 0 : baie), rayon de la ligne de courant r = 1/κ + d :
        ζ = u (1/Rᵢ − κ/(1 + κ d)),   ζ/f = (u0/c) exp(-d/Rᵢ) (1 − Rᵢ κ/(1 + κ d)).
    Pour κ = 0 : ζ/f = u0/c (cisaillement cyclonique de l'onde). Au bout d'un cap étroit (R < Rᵢ), la
    courbure l'emporte et ζ devient anticyclonique pour u0 > 0 (crête), cyclonique pour u0 < 0 (creux).
    NaN si 1 + κ d ≤ 0.2 (au-delà du centre de courbure d'une baie).
    """
    Ri = c / f
    d = np.asarray(d, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    den = 1.0 + kappa * d
    keff = np.where(den > 0.2, kappa / np.where(den > 0.2, den, 1.0), np.nan)
    return (u0 / c) * np.exp(-d / Ri) * (1.0 - Ri * keff)


# ------------------------------------------------------------------ régimes d'échelle
# Seuils indicatifs : à calibrer. Ke = R_max/Rᵢ, Ro = V_max/(f R_max) (R_max : rayon de la vitesse maximale).
RO_SUB = 0.3      # au-dessus : agéostrophique (sous-mésoéchelle)
RO_SMALL = 3.0    # au-dessus : la rotation devient secondaire
KE_BASIN = 3.0    # Ke au-dessus duquel la structure est à l'échelle du bassin
KE_MESO = 0.5     # Ke en dessous duquel L ≪ Rᵢ


def scale_regime(Ke, Ro):
    """Régime d'échelle d'une structure en rotation, défini par la dynamique (Ke, Ro) et non par une taille.

    Dans l'océan la mésoéchelle fait 10–100 km parce que Rᵢ y vaut 30–50 km ; ici Rᵢ ≈ 3 km (été) et ≈ 1–2 km
    (hiver) : la même structure peut changer de régime avec la saison.
    """
    Ro = abs(Ro)
    if Ro >= RO_SMALL:
        return "petite échelle (Ro ≳ 3, rotation secondaire)"
    if Ro >= RO_SUB:
        return "sous-mésoéchelle (Ro ≈ 0,3–3, agéostrophique : fronts, filaments, tourbillons côtiers)"
    if Ke >= KE_BASIN:
        return "grande échelle (bassin, Ke ≳ 3, quasi-géostrophique barotrope)"
    if Ke >= KE_MESO:
        return "mésoéchelle (Ke ≈ 0,5–3, Ro < 0,3, quasi-géostrophique barocline)"
    return "petite échelle quasi-géostrophique (L ≪ Rᵢ, Ro < 0,3, confiné près de la surface)"


# ------------------------------------------------------------ discrétisation d'un tourbillon
def vortex_zeta_discretization(R_over_dx, offset=0.0):
    """Erreur *cinématique* sur ζ d'un tourbillon gaussien V = V0 (r/R) exp((1−(r/R)²)/2) échantillonné sur une
    grille de pas 1, ζ calculée par différences centrées sur le champ de vitesse exact.

    Retourne (ζ_max numérique / ζ_max exact, erreur maximale relative sur le profil). `offset` décale le centre
    (en mailles). N'inclut PAS l'amortissement par le schéma d'advection ni la viscosité du modèle.
    Pour R > 30 mailles : asymptote 0,5/R² (ajustée sur R = 4–12), sans calcul sur grille.
    """
    R = float(R_over_dx)
    if R > 30.0:
        e = 0.5 / R**2
        return 1.0 - e, e
    n = int(max(81, 8 * R)) | 1
    xs = np.arange(n) - n // 2 + offset
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y) + 1e-12
    x = r / R
    V = x * np.exp((1 - x**2) / 2)
    u, v = -V * Y / r, V * X / r
    zeta = np.gradient(v, axis=1) - np.gradient(u, axis=0)
    zex = (1.0 / R) * (2 - x**2) * np.exp((1 - x**2) / 2)
    return float(zeta.max() / zex.max()), float(np.abs(zeta - zex).max() / zex.max())


def gradient_response(lam_over_dx):
    """Rapport (pente numérique / pente exacte) d'une onde sin(kx) par différence centrée d'ordre 2 ;
    0 pour λ ≤ 2Δx (limite de Nyquist)."""
    lam = np.asarray(lam_over_dx, dtype=float)
    k = 2 * np.pi / np.maximum(lam, 1e-9)
    return np.where(lam > 2.0, np.sin(k) / k, 0.0)
