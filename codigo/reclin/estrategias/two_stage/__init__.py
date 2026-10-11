"""Two-Stage: detecção de relação seguida da classificação do tipo.

Dois classificadores binários independentes, cada um com o seu próprio
fine-tuning a partir dos pesos ORIGINAIS do BioBERTpt (`pucpr/biobertpt-all`,
numa revisão fixada por `reclin.pretreino`). Nenhum estágio parte de um
checkpoint ajustado — nem do baseline, nem do outro estágio, nem de outra
estratégia (`inicializacao`).

* Estágio 1 — existe relação? `relacao` × `no_relation`, treinado com todos os
  pares candidatos do TRAIN (152.686: 8.351 `relacao`, 144.335 `no_relation`).
  Checkpoint escolhido pelo maior AP da classe `relacao` no DEV completo.
* Estágio 2 — qual o tipo? `negation_of` × `associated_with`, treinado só com
  os pares do TRAIN que têm relação anotada (8.351: 1.255 e 7.096). Checkpoint
  escolhido pelo maior F1 de `negation_of` nos 913 pares do DEV com relação
  anotada (argmax).
* Inferência (`composicao`): um par vai ao estágio 2 quando P(relacao),
  arredondada a 4 casas (a convenção dos sidecars), é >= τ; os demais ficam
  `no_relation`; o estágio 2 decide o tipo pelo argmax. A saída tem os 3
  rótulos da tarefa sobre o conjunto completo. τ é escolhido no DEV pelo
  pipeline completo (`avaliacao.limiar`) e congelado antes do TEST.

Os dois estágios usam as mesmas janelas e os mesmos marcadores do baseline
(`entrada`), a mesma configuração base (`config.Config`: semente, lotes, `lr`,
pesos `balanced` — aqui calculados no TRAIN de cada estágio) e o mesmo laço de
treino (`treino.laco`), com retomada exata. Escolher cada estágio pela sua
própria métrica não garante a melhor combinação para o pipeline completo; não
há otimização conjunta de épocas no DEV.

Os dois estágios só se encontram na composição, que é inferência: o estágio 2
treina com os rótulos anotados, nunca com predições do estágio 1.
"""
from __future__ import annotations

from dataclasses import dataclass

from reclin.config import Config
from reclin.tarefa import ASSOC, NEG, NOREL

NOME = "two_stage"
ESTAGIOS = (1, 2)
ESTRATEGIA_ESTAGIO = {1: "two_stage_e1", 2: "two_stage_e2"}

# Rótulos de cada estágio, na ordem dos ids da cabeça.
ROTULOS = {1: ("relacao", "no_relation"), 2: ("negation_of", "associated_with")}
RELACAO, SEM_RELACAO = 0, 1             # estágio 1
NEG_E2, ASSOC_E2 = 0, 1                 # estágio 2

# Critério de seleção do checkpoint de cada estágio (chave do histórico do DEV).
CAMPO_SELECAO = {1: "dev_ap_relacao", 2: "dev_negation_of_f1_relacoes_anotadas"}
DESCRICAO_SELECAO = {
    1: "maior AP da classe relacao no DEV completo; empate -> a primeira época",
    2: "maior F1 de negation_of nos pares do DEV com relação anotada (argmax); empate -> a primeira época",
}
DADOS_TREINO = {
    1: "todos os pares candidatos do TRAIN (negation_of e associated_with -> relacao)",
    2: "só os pares do TRAIN com relação anotada (negation_of ou associated_with)",
}
CASAS_ROTEAMENTO = 4                    # P(relacao) arredondada como nos sidecars


@dataclass(frozen=True)
class ConfigEstagio1(Config):
    """Estágio 1: o teto de épocas do baseline (mesmos dados, mesmos passos)."""
    epochs: int = 3


@dataclass(frozen=True)
class ConfigEstagio2(Config):
    """Estágio 2: 10 épocas (131 passos por época, 1.310 no total)."""
    epochs: int = 10


CONFIGS = {1: ConfigEstagio1, 2: ConfigEstagio2}


def rotulo_estagio1(rotulo: int) -> int:
    """Rótulo da tarefa (3 classes) → estágio 1."""
    return SEM_RELACAO if rotulo == NOREL else RELACAO


def rotulo_estagio2(rotulo: int) -> int:
    """Rótulo da tarefa (só negation_of ou associated_with) → estágio 2."""
    if rotulo == NEG:
        return NEG_E2
    if rotulo == ASSOC:
        return ASSOC_E2
    raise ValueError("o estágio 2 só recebe pares com relação anotada")


def nome_execucao(estagio: int, config: Config) -> str:
    return f"twostage_e{estagio}_{config.encoder}_seed{config.seed}"


def nome_composicao(config: Config) -> str:
    return f"twostage_{config.encoder}_seed{config.seed}"


PREFIXO_VALIDACAO = {1: "twostage_e1", 2: "twostage_e2"}
NOME_COMPOSICAO_VALIDACAO = "twostage_validacao"
