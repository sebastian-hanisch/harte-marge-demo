"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import hmg_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None  # erlaubte Werte (Auswahl, Optionsregler); sonst gelten lo und hi


def _bool(v):
    return str(v).lower() in ("1", "true", "ja", "yes")


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "n1_slider": SettingSpec("n1", int, C.DEFAULT_N1, C.N1_MIN, C.N1_MAX),
    "puffer_select": SettingSpec("pf", float, C.DEFAULT_PUFFER, choices=C.PUFFER_OPTIONS),
    "theta_off": SettingSpec("th", int, C.DEFAULT_THETA, C.THETA_MIN, C.THETA_MAX),
    "perc_k": SettingSpec("pk", int, C.DEFAULT_PERC, C.PERC_MIN, C.PERC_MAX),
    "e2_set": SettingSpec("e2", str, C.DEFAULT_E2, choices=C.E2_SETS),
    "e3_data": SettingSpec("e3", str, C.DEFAULT_E3, choices=C.E3_SETS),
    "e3_point": SettingSpec("e3p", int, C.DEFAULT_E3_POINT, C.E3_POINT_MIN, C.E3_POINT_MAX),
    "e3_dist": SettingSpec("e3d", float, C.DEFAULT_E3_DIST, choices=C.E3_DIST_OPTIONS),
    "e3_pos": SettingSpec("e3x", float, C.DEFAULT_E3_POS, choices=C.E3_POS_OPTIONS),
    "e4_abstand": SettingSpec("e4a", float, C.DEFAULT_E4_ABSTAND, choices=C.E4_ABSTAND_OPTIONS),
    "e4_n": SettingSpec("e4n", int, C.DEFAULT_E4_N, C.E4_N_MIN, C.E4_N_MAX),
    "e4_c": SettingSpec("e4c", int, C.DEFAULT_E4_C, choices=C.E4_C_OPTIONS),
    "e4_tol": SettingSpec("e4t", _bool, False, choices=(False, True)),
    "e4_steps": SettingSpec("e4s", int, C.DEFAULT_E4_STEPS, choices=C.E4_STEPS_OPTIONS),
    "kern_choice": SettingSpec("kern", str, C.DEFAULT_KERN, choices=C.KERN_CHOICES),
    "e5_data": SettingSpec("e5", str, C.DEFAULT_E5, choices=C.E5_SETS),
}
PRESET_KEYS = {
    "seed": "seed_input", "n1": "n1_slider", "puffer": "puffer_select", "theta": "theta_off", "perc": "perc_k", "e2": "e2_set", "e3": "e3_data", "e3_point": "e3_point",
    "e3_dist": "e3_dist", "e3_pos": "e3_pos", "e4_abstand": "e4_abstand", "e4_n": "e4_n", "e4_c": "e4_c", "e4_tol": "e4_tol", "e4_steps": "e4_steps", "kern": "kern_choice", "e5": "e5_data",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean(spec: SettingSpec, value):
    """Wert auf den erlaubten Bereich begrenzen (Permalink); ungültige Auswahl gibt den Standardwert."""
    if spec.choices is not None:
        return value if value in spec.choices else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = clean(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt alle Regler: die im Preset genannten auf ihren Wert, alle übrigen auf den Standardwert."""
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = preset.get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf
    (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (jeder Knopf setzt alle Regler; die Hilfe nennt das Experiment):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def center_theta():
    """Knopf „Richtung der größten Marge“: Abweichung 0."""
    st.session_state["theta_off"] = 0
