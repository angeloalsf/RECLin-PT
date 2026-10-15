"""Variante de encoder (biobertpt-clin) no Two-Stage: separação de identidade, pesos e nomes. Sem treino nem rede."""
from reclin.config import ENCODERS
from reclin.estrategias import two_stage
from reclin.experimento import apresentacao, operacoes


def _espec(encoder):
    return operacoes.Especificacao(estrategia="two_stage_e1", script="two_stage.py", config_cls=two_stage.CONFIGS[1],
                                   estagio=1, encoder=encoder)


def test_variante_tem_modelo_proprio_e_sem_nome_oficial():
    base, clin = _espec("biobertpt"), _espec("biobertpt_clin")
    assert base.config_protocolo().modelo == ENCODERS["biobertpt"] == "pucpr/biobertpt-all"
    assert clin.config_protocolo().modelo == ENCODERS["biobertpt_clin"] == "pucpr/biobertpt-clin"
    assert clin.alias_oficial is None
    assert base.config_protocolo() != clin.config_protocolo()


def test_pasta_dos_pesos_por_modelo():
    assert apresentacao.pasta_dos_pesos("/p", "abc").name == "biobertpt-all@abc"
    assert apresentacao.pasta_dos_pesos("/p", "abc", "biobertpt-clin").name == "biobertpt-clin@abc"


def test_execucao_atual_ignora_outro_encoder(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    ctx = SimpleNamespace(espec=SimpleNamespace(estrategia="baseline", encoder="bertimbau"),
                          raiz_local=tmp_path, raiz_drive=None)
    registros = {"b_all": {"parametros_de_treinamento": {"encoder": "biobertpt"}, "atualizado_em": "2026-03-01"},
                 "b_tim": {"parametros_de_treinamento": {"encoder": "bertimbau"}, "atualizado_em": "2026-01-01"}}
    monkeypatch.setattr(apresentacao, "_exigir_estrategia", lambda c: None)
    monkeypatch.setattr(apresentacao.operacoes, "ids_estrategia", lambda r: "baseline")
    monkeypatch.setattr(apresentacao.plano, "execucoes", lambda *a: [
        {"nome": n, "run_id": n, "onde": "local"} for n in registros])
    monkeypatch.setattr(apresentacao.proveniencia, "ler", lambda p: registros[Path(p).name])
    (tmp_path / "b_tim" / "x").mkdir(parents=True)                       # só para a restauração não ser pedida
    monkeypatch.setattr(apresentacao.proveniencia, "ARQUIVO", "x")
    assert apresentacao.execucao_atual(ctx) == "b_tim"                   # a mais recente é a do biobertpt, mas é outro encoder
    ctx.espec.encoder = "biobertpt_clin"
    with pytest.raises(apresentacao.ErroOperacao, match="nenhuma execução do encoder"):
        apresentacao.execucao_atual(ctx)
