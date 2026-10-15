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
