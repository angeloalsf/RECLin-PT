"""`associated_with` sem direção: a fusão dos dois sentidos de cada par."""
from reclin.analises.associated_sem_direcao import avaliar
from reclin.tarefa import ASSOC as A
from reclin.tarefa import LABELS
from reclin.tarefa import NEG as N
from reclin.tarefa import NOREL as O

ASSOC, NEG, NOREL = LABELS[A], LABELS[N], LABELS[O]


def _c(doc, e1, e2, label):
    return {"doc_id": doc, "e1": {"id": e1}, "e2": {"id": e2}, "label": label}


def test_erro_so_de_direcao_deixa_de_ser_erro():
    cands = [_c("d", "T1", "T2", ASSOC), _c("d", "T2", "T1", NOREL),     # anotado T1->T2
             _c("d", "T3", "T4", NOREL), _c("d", "T4", "T3", NOREL),
             _c("d", "T5", "T6", NEG), _c("d", "T6", "T5", NOREL)]
    r = avaliar(cands, [O, A, O, O, N, O])                              # previu só o sentido inverso
    assert (r["com_direcao"]["tp"], r["com_direcao"]["fp"], r["com_direcao"]["fn"]) == (0, 1, 1)
    assert (r["sem_direcao"]["tp"], r["sem_direcao"]["fp"], r["sem_direcao"]["fn"]) == (1, 0, 0)
    assert r["sem_direcao"]["n_pares_nao_ordenados"] == 3 and r["sem_direcao"]["suporte"] == 1
    assert r["sem_direcao"]["pares_sem_o_sentido_inverso"] == 0 and r["diferenca_f1"] == 1.0


def test_dois_sentidos_previstos_contam_uma_vez_e_erro_real_continua_erro():
    cands = [_c("d", "T1", "T2", ASSOC), _c("d", "T2", "T1", NOREL),
             _c("d", "T3", "T4", NOREL), _c("d", "T4", "T3", NOREL),
             _c("e", "T1", "T2", ASSOC)]                                # sem o inverso entre os candidatos
    r = avaliar(cands, [A, A, A, O, O])
    assert (r["com_direcao"]["tp"], r["com_direcao"]["fp"], r["com_direcao"]["fn"]) == (1, 2, 1)
    assert (r["sem_direcao"]["tp"], r["sem_direcao"]["fp"], r["sem_direcao"]["fn"]) == (1, 1, 1)
    assert r["sem_direcao"]["pares_sem_o_sentido_inverso"] == 1
