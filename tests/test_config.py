from clausewitz.config import Settings


def test_model_ids_come_from_env(monkeypatch):
    monkeypatch.setenv("GENERATE_MODEL", "gen-x")
    monkeypatch.setenv("VERIFY_MODEL", "ver-y")
    s = Settings(_env_file=None)
    assert (s.generate_model, s.verify_model) == ("gen-x", "ver-y")


def test_model_ids_default_empty(monkeypatch):
    monkeypatch.delenv("GENERATE_MODEL", raising=False)
    assert Settings(_env_file=None).generate_model == ""
