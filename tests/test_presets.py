"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ein eigener Zustand; Permalink begrenzt."""

import hmg_constants as C
import hmg_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_for_their_regler():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert P.clean(spec, value) == value, (name, key, value)  # innerhalb der Grenzen bzw. unter den erlaubten Werten


def test_default_preset_equals_the_default_state_and_all_presets_are_distinct_states():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {}
    states = {name: tuple(sorted((k, v) for k, v in p.items())) for name, p in C.PRESETS.items()}
    assert len(set(states.values())) == len(states)  # kein Preset ist von einem anderen nicht unterscheidbar


def test_setting_specs_cover_every_preset_key_and_every_state_key_has_a_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for state_key, spec in P.SETTING_SPECS.items():
        assert P.clean(spec, spec.default) == spec.default, state_key
        assert spec.url_param and spec.caster


def test_permalink_value_is_clamped_or_replaced_by_the_default():
    spec = P.SETTING_SPECS["n1_slider"]
    assert P.clean(spec, 10 ** 6) == C.N1_MAX and P.clean(spec, -5) == C.N1_MIN
    assert P.clean(P.SETTING_SPECS["kern_choice"], "unbekannt") == C.DEFAULT_KERN
    assert P.clean(P.SETTING_SPECS["puffer_select"], 0.123) == C.DEFAULT_PUFFER
    assert P.SETTING_SPECS["e4_tol"].caster("True") is True and P.SETTING_SPECS["e4_tol"].caster("False") is False


def test_url_params_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_option_lists_are_ordered_and_contain_their_defaults():
    assert C.DEFAULT_PUFFER in C.PUFFER_OPTIONS and list(C.PUFFER_OPTIONS) == sorted(C.PUFFER_OPTIONS)
    assert C.DEFAULT_E3_DIST in C.E3_DIST_OPTIONS and C.DEFAULT_E3_POS in C.E3_POS_OPTIONS and C.DEFAULT_E4_C in C.E4_C_OPTIONS and C.DEFAULT_E4_STEPS in C.E4_STEPS_OPTIONS
    assert C.DEFAULT_E4_ABSTAND in C.E4_ABSTAND_OPTIONS and C.DEFAULT_KERN in C.KERN_CHOICES and C.DEFAULT_E2 in C.E2_SETS and C.DEFAULT_E5 in C.E5_SETS
    assert 1e6 not in C.E4_C_OPTIONS  # C = 1e6 gibt es nur in der eingefrorenen Tabelle (kein Reglerwert)
    assert max(C.E4_STEPS_OPTIONS) <= C.HARD_MAX_STEPS == 200000 and C.E2_RANDOM_N <= 8 and C.E4_N_MAX <= 40 and C.N1_MAX <= 40
