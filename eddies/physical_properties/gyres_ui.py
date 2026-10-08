"""Affichage interactif robuste pour les notebooks à curseurs.

`live_output(func, controls)` remplace `ipywidgets.interactive_output`. À chaque changement d'un curseur,
`func(**valeurs)` est appelée ; ses tableaux (Markdown) et ses figures (PNG) sont **capturés** puis placés dans
des widgets `HTML` et `Image` dont le contenu **remplace** le précédent.

Pourquoi : avec `interactive_output`, l'affichage passe par un widget `Output` et `clear_output(wait=True)`.
Constats faits sur ce projet :
- l'utilisateur a vu plusieurs copies du même tableau et de la même figure (interface non précisée) ;
- dans JupyterLab 4.4, un widget `Output` fait disparaître la 3e figure de `profondeur_cote_rugosite.ipynb`
  (3 figures et 4 tableaux avec un appel direct de la fonction, 2 figures et 4 tableaux avec `Output`),
  avec l'ancien comme avec le nouveau code d'appel.
Avec des widgets dont on remplace la valeur, il ne peut rester ni copie ni figure perdue : chaque redessin
reconstruit la liste complète des enfants.

Autres différences :
- un redessin est ignoré si les valeurs des curseurs n'ont pas changé (aller-retour parasite d'un curseur) ;
- ré-exécuter la cellule qui appelle `live_output` détache les anciens observateurs (même `key`).

Dépendance optionnelle : `mistune` (déjà présent avec nbconvert) pour convertir les tableaux Markdown en HTML ;
sans lui, le Markdown est affiché en texte brut.

Variable d'environnement GYRES_UI_MODE : `live` (défaut), `static` (un seul appel de `func` aux valeurs
actuelles, affichage dans la sortie de la cellule : pour exécuter un notebook sans interface, voir
`check_notebooks.py`) ou `none` (aucun appel, pour les tests de fonctions).
"""
from __future__ import annotations

import base64
import html
import os
import traceback

import numpy as np
import ipywidgets as w
from IPython.utils.capture import capture_output

_REGISTRY: dict = {}

_CSS = ("<style>.gyres-md table{border-collapse:collapse;margin:4px 0 10px 0}"
        ".gyres-md th,.gyres-md td{border:1px solid rgba(128,128,128,.45);padding:2px 9px;text-align:left}"
        ".gyres-md th{background:rgba(128,128,128,.12)}.gyres-md p{margin:4px 0}</style>")


def _md_to_html(md: str) -> str:
    try:
        import mistune
        body = mistune.create_markdown(plugins=["table"])(md)
    except Exception:  # mistune absent ou ancien : texte brut
        body = "<pre>" + html.escape(md) + "</pre>"
    return f'<div class="gyres-md">{_CSS}{body}</div>'


def _to_widgets(cap) -> list:
    """Convertit une sortie capturée (IPython.utils.capture) en widgets, dans l'ordre d'affichage."""
    children = []
    text = (cap.stdout or "") + (cap.stderr or "")
    for o in cap.outputs:
        d = o.data
        if "image/png" in d:
            raw = d["image/png"]
            raw = base64.b64decode(raw) if isinstance(raw, str) else raw
            children.append(w.Image(value=raw, format="png", layout=w.Layout(max_width="100%", height="auto")))
        elif "text/markdown" in d:
            children.append(w.HTML(_md_to_html(d["text/markdown"])))
        elif "text/html" in d:
            children.append(w.HTML(d["text/html"]))
        elif "text/plain" in d:
            children.append(w.HTML("<pre>" + html.escape(d["text/plain"]) + "</pre>"))
    if text.strip():
        children.append(w.HTML("<pre>" + html.escape(text) + "</pre>"))
    return children


def _same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not np.isclose(x, y, rtol=1e-9, atol=0.0):
                return False
        elif x != y:
            return False
    return True


def live_output(func, controls: dict, key: str | None = None):
    """Relie `controls` (dict nom -> widget) à `func(**valeurs)` ; retourne le widget à afficher."""
    mode = os.environ.get("GYRES_UI_MODE", "live")
    if mode == "none":
        return w.VBox()
    if mode == "static":
        func(**{k: v.value for k, v in controls.items()})
        return w.VBox()

    key = key or func.__name__
    for wd, obs in _REGISTRY.pop(key, []):
        wd.unobserve(obs, names="value")

    holder = w.VBox()
    last: dict = {}

    def render(*_):
        kw = {k: v.value for k, v in controls.items()}
        if "kw" in last and _same(kw, last["kw"]):
            return
        last["kw"] = kw
        with capture_output() as cap:
            try:
                func(**kw)
            except Exception:
                traceback.print_exc()
        holder.children = _to_widgets(cap)

    for wd in controls.values():
        wd.observe(render, names="value")
    _REGISTRY[key] = [(wd, render) for wd in controls.values()]
    render()
    return holder
