"""Especialização do RECLin-PT em `negation_of`.

O conhecimento sobre como a negação aparece no SemClinBr que qualquer
estratégia pode usar ou ignorar: o léxico de pistas induzido do TRAIN e o
predicado `e_pista` (`lexico`). Continuaria valendo se todas as estratégias
atuais fossem descartadas.

O que é estratégia não fica aqui: o filtro de pistas, a regra pura, a
calibração de `min_freq` e a seleção de candidatos do fine-tuning restrito
ficam em `reclin/estrategias/`. O léxico congelado é especialização; o
procedimento que o escolheu pertence à estratégia de filtro.

Depende só do núcleo (`tarefa`).
"""
