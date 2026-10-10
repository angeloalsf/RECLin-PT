"""O que uma estratégia treinada informa ao orquestrador do treino.

O orquestrador (`treino.classificador`) faz sempre o mesmo: lê TRAIN e DEV,
monta os exemplos, o tokenizer e o modelo, treina com `treino.laco`, grava a
execução e, depois, avalia o TEST em outra operação. O que muda de uma
estratégia para outra cabe numa `Montagem`:

* `estrategia` — o nome gravado na execução (e parte da sua identidade:
  uma execução não é retomada por outra estratégia);
* `classe_config` — a configuração da estratégia (derivada de `config.Config`);
* `criar_modelo(modelo_id, tokenizer, config)` — o modelo; o padrão é o
  classificador de sequência ([CLS]) de `modelos`; `classe_modelo` é a classe
  usada para recarregar o `melhor_modelo/` (None = o classificador);
* `selecionar(candidatos)` — as posições dos candidatos usados para treinar,
  escolher a época e prever (None = todos). Com seleção, o DEV é pontuado
  depois de remapeado ao conjunto completo (`execucao.subconjunto`) e as
  métricas do subconjunto entram no histórico como `dev_restrito_*`;
* `registro` — a configuração própria da estratégia (léxico, espaço, cabeça),
  gravada em `config.json` e conferida na retomada;
* `extras_sidecar` — campos que a estratégia acrescenta aos sidecars, antes
  dos do orquestrador (o legado gravava `espaco`, `cabeca`, `encoder`);
* `descrever(tokenizer, modelo, exemplos)` — informação medida antes do treino
  (ou do TEST) que vai para `treino.json` (os marcadores ausentes da
  Pair-Aware, por exemplo). Não pode consumir geradores aleatórios.

`CLASSIFICADOR` é a montagem do classificador da tarefa (etapa 5), e o
baseline é ela com o nome da estratégia.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from reclin import modelos
from reclin.config import Config


def _classificador(modelo_id: str, tokenizer: Any, config: Config) -> Any:
    return modelos.carregar_classificador(modelo_id, tokenizer)


@dataclass(frozen=True)
class Montagem:
    estrategia: str
    classe_config: type = Config
    criar_modelo: Callable[[str, Any, Any], Any] = _classificador
    classe_modelo: Any = None
    selecionar: Callable[[Sequence[dict]], list[int]] | None = None
    registro: dict[str, Any] = field(default_factory=dict)
    extras_sidecar: dict[str, Any] = field(default_factory=dict)
    descrever: Callable[[Any, Any, dict], dict[str, Any]] | None = None


CLASSIFICADOR = Montagem(estrategia="classificador")
