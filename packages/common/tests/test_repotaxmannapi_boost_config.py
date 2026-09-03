from common.repotaxmannapi_boost_config import load_repotaxmannapi_boost_config


def test_loads_latest_edition_years():
    config = load_repotaxmannapi_boost_config()
    assert config["latest_edition_years"] == {
        "111050000000010687": "2026",
        "111050000000020042": "2026",
        "111050000000010622": "2017",
    }


def test_loads_gst_tariff_current_editions():
    config = load_repotaxmannapi_boost_config()
    assert config["gst_tariff"] == {
        "goods_subgroup_id": "111050000000017647",
        "goods_latest_subsubgroup_id": "111050000000020114",
        "services_subgroup_id": "111050000000017648",
        "services_latest_subsubgroup_id": "111050000000020066",
        "cgst_sgst_subgroup_id": "111050000000018741",
        "cgst_sgst_latest_subsubgroup_id": "111050000000020065",
    }


def test_loads_forms_config():
    config = load_repotaxmannapi_boost_config()
    assert config["forms"] == {"formtype_id": "frmtyp002", "latest_year": "2026"}


def test_strips_comment_keys_at_every_level():
    """"_comment" keys are inline documentation only - never a real value a caller should
    see, at the top level or nested inside any sub-object."""
    config = load_repotaxmannapi_boost_config()
    assert "_comment" not in config
    for value in config.values():
        if isinstance(value, dict):
            assert "_comment" not in value


def test_loading_twice_returns_the_same_cached_object():
    assert load_repotaxmannapi_boost_config() is load_repotaxmannapi_boost_config()
