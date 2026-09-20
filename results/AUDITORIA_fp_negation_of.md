# Auditoria dos falsos positivos remanescentes de `negation_of`

`20/09/2026` · `split = test` · `execucao sorteada = filtro_biobertpt_seed43` · **relatorio, nao decisao**

Nada foi filtrado, editado ou reclassificado com base nesta leitura. Nenhum
numero do Cap. 6 depende deste arquivo. O produto e uma fracao, e ela serve para
saber se o teto que resta perseguir e 0,95 ou 0,88 antes de gastar GPU atras
dele.

## Sorteio

`random.Random(20260920).choice(...)` sobre as quatro execucoes filtradas,
com a semente fixada no codigo antes do primeiro sorteio. Saiu **filtro_biobertpt_seed43**.
Nao houve segunda tentativa.

## Criterio, fixado antes de olhar os casos

Seja `G` o conjunto de alvos que o gold liga a essa pista nesse documento.

| # | condicao | classe |
|---|---|---|
| 1 | `G` vazio e `gap <= 1` | provavel erro de anotacao |
| 2 | `G` vazio e `gap > 1` | erro do modelo |
| 3 | `G` nao vazio, so separadores de coordenacao entre o alvo anotado e `e2` | ambiguidade genuina |
| 4 | `G` nao vazio e `gap` menor ou igual ao do alvo anotado mais proximo | ambiguidade genuina |
| 5 | resto | erro do modelo |

As regras sao aplicadas de cima para baixo. O criterio e mecanico de proposito,
para que rode igual para quem repetir e nao dependa de os casos terem sido lidos
antes de ele ser escrito.

### Correcao de implementacao, no mesmo dia

A primeira execucao deste script tinha um defeito na regra 3, e o defeito
aparece ao ler a saida. Ela pegava o trecho entre as entidades fatiando
`doc["text"]` com os offsets de `doc["entities"]`, e no SemClinBr esses offsets
nao indexam o texto: so 64,4% das entidades do test satisfazem
`text[start:end] == texto_da_entidade`. A deriva e cumulativa e negativa dentro
do documento, o que e a assinatura de uma normalizacao de espaco em branco feita
depois do calculo dos offsets. Com isso a regra 3 nunca disparava, e listas
coordenadas obvias como `NEGA CIRURGIAS E TRAUMAS OCULARES` caiam na regra 5.

A correcao reancora cada entidade procurando a propria forma de superficie a
partir da posicao esperada, o que recupera 97,1% das entidades do test. Ela vale
so para LER o trecho entre as entidades e imprimir o contexto. O `gap` continua
saindo dos offsets originais, que sao mutuamente consistentes e sao os que
`src/candidates.py` usa, para a auditoria falar do mesmo objeto que o sistema
auditado. O criterio nao mudou, so a leitura do texto que ele consulta.

## Resultado

A execucao sorteada acerta 137 das 152 negacoes do teste, perde 15 e
produz 33 falsos positivos.

| classe | casos | fracao dos FP |
|---|---|---|
| erro do modelo | 9 | 27,3% |
| ambiguidade genuina | 6 | 18,2% |
| provavel erro de anotacao do gold | 18 | 54,5% |
| **total** | **33** | |

Pelo criterio acima, **24 dos 33 FP (72,7%) nao sao erro de leitura clinica**.
Somando os como acertos, a precisao da execucao sorteada iria de 0,8059 para 0,9384, e o F1 de 
0,8509 para 0,9195.
Esse segundo numero **nao e um resultado**. Ele e o teto que sobra se toda a
divergencia de anotacao fosse resolvida a favor do modelo, e serve so para
dimensionar quanto ainda ha para ganhar em precisao.

## Casos, um a um

### erro do modelo (9)

| doc | pista `e1` | alvo `e2` | gap | gold do par | alvos anotados da pista | regra | trecho |
|---|---|---|---|---|---|---|---|
| 8965 | Nega | queixas | 18 | no_relation | algias | 5 | ...al Philadelphia já solicitado pelos médicos. Nega algias ou outras queixas. Boa aceitação alimentar. Aguarda realização... |
| 9336 | NÃO | MANOBRAS DE RCP | 16 | no_relation | (nenhum) | 2 | ...ARRITMIA >> NOVA PCR EM FV >> ASSISTOLIA , NÃO RESPONDENDO ÀS MANOBRAS DE RCP , DURAÇÃO DE 45 MINUTOS . ÓBITO ÀS 12H48MIN... |
| 9570 | não | medicações | 16 | no_relation | (nenhum) | 2 | ...ou há 12 anos, CT alta # Em uso de marevan - não sabe as outras medicações que usa Lab 26/01/2013 - Cr 1,2 / Glic 89/ U... |
| 9593 | NEGA | PALPITAÇÕES | 13 | no_relation | SINTOMAS | 5 | ...ERE ESTAR SE SENTINDO BEM COM AS MEDICAÇÕES. NEGA SINTOMAS DE PALPITAÇÕES. RELATA DISPNEIA AOS GRANDES ESFORCOS, COM M... |
| 9625 | Sem | família | 20 | no_relation | cancer | 5 | ...emas cardíacos (?). Pai, 90anos, hipertenso. Sem casos de cancer na família. Hipertensa há 15 anos. Hipotireoidismo; ref... |
| 9888 | NEGA | NEOPLASIA | 22 | no_relation | (nenhum) | 2 | ...CERCA DE 1-2 ANOS. AM: NEGA PATOLOGIAS. AF: NEGA HISTORIA FAMILIAR DE NEOPLASIA. NAO TRAZ EXAMES COMPLEMENTARES. AO EXAME: P... |
| 9904 | NEGA | TRATANDO | 23 | no_relation | HAS, DM | 5 | ...SE DE QUADRIL A ESQUERDA REALIZADA EM 2007. NEGA HAS / DM, REFRE ESTRA TRATANDO PARA HIPOTIREOIDISMO, PACIENTE OBESA PACIEN... |
| 9904 | NEGA | FLEXO-EXTENSÃO | 7 | no_relation | (nenhum) | 2 | ...M DIFICULDADE COLUNA APARENTEMENTE NO EIXO NEGA DOR A FLEXO-EXTENSÃO E ROTAÇÕES SEM DOR A PALPAÇAO EM COLUNA LOM... |
| 9907 | NEGA | DOENÇA RENAL | 13 | no_relation | (nenhum) | 2 | ...40 ANOS MAÇO, PAROU HÁ 5 ANOS - SEDENTÁRIA - NEGA HISTÓRIA DE DOENÇA RENAL NA FAMÍLIA TROUXE EXAMES LABORATORIAIS: - H... |

### ambiguidade genuina (6)

| doc | pista `e1` | alvo `e2` | gap | gold do par | alvos anotados da pista | regra | trecho |
|---|---|---|---|---|---|---|---|
| 9116 | NEGA | RESPONSIVA AOS REFLEXOS | 11 | no_relation | CEFALEIA | 3 | ...TIVA, ORINTADA NO TEMPO E NO ESPAÇO, CORADA. NEGA CEFALEIA, RESPONSIVA AOS REFLEXOS. ACESSO VENOSO EM FOSSA CUBITAL D SEM ALTERA... |
| 9298 | evacuação | após | 9 | no_relation | ausente | 3 | ...eg umbilical. nega algia. diurese presente e evacuação ausente após cirurgia. corado, abdome flácido. avp permeá... |
| 9544 | NEGA | EDEMA | 17 | no_relation | DOR PRECORDIAL | 3 | ...S # RELATA DISPNEIA PARA MODERADOS ESFORÇOS. NEGA DOR PRECORDIAL, EDEMA DE MMII, ORTOPNEIA E DPN. - CINTILO (06/10/... |
| 9774 | NEGA | TRAUMAS OCULARES | 13 | no_relation | CIRURGIAS | 3 | ...AL, HA 2 ANOS, NAO SABE REFERIR BEM A CAUSA. NEGA CIRURGIAS E TRAUMAS OCULARES. NEGA HF DE GLAUCOMA E CEGUEIRA. AV: 20/25... |
| 9844 | sem | sinais de isquemia miocárdica | 8 | no_relation | dor | 3 | ...rcorrências. Evoluiu bem após procedimento, sem dor ou sinais de isquemia miocárdica. Recebe alta em excelente estado geral, sin... |
| 9904 | NEGA | DOR INCIDIOSA | 11 | no_relation | TRAUMA | 3 | ...DO LAR DOR EM COLUNA LOMBAR DE LONGA DATA, NEGA TRAUMA DOR INCIDIOSA, NEGA USO DE ANALGESICO JÁ REALIZOU FST, NO... |

### provavel erro de anotacao do gold (18)

| doc | pista `e1` | alvo `e2` | gap | gold do par | alvos anotados da pista | regra | trecho |
|---|---|---|---|---|---|---|---|
| 8940 | SEM | PRESENÇA DE GRUMOS | 1 | no_relation | (nenhum) | 1 | ...DIA QUANTIDADE, SVD COM DIURESE CONCENTRADA, SEM PRESENÇA DE GRUMOS. ELIMINAÇÕES INTESTINAIS AUSENTES ATÉ O MOME... |
| 8940 | AUSENTES | ELIMINAÇÕES INTESTINAIS | 1 | no_relation | (nenhum) | 1 | ...DIURESE CONCENTRADA, SEM PRESENÇA DE GRUMOS. ELIMINAÇÕES INTESTINAIS AUSENTES ATÉ O MOMENTO, REALIZADO MEDIDAS E HIGIENE E... |
| 9000 | sem | prescriçao medica | 1 | no_relation | (nenhum) | 1 | ...egue cuidados. Recebido plantao com paciente sem prescriçao medica, entrado em contato com plantao da ortopedia... |
| 9055 | S/ | RA | 1 | no_relation | (nenhum) | 1 | ...ar ambiente com ausculta pulmonar com MV +, S/ RA, saturando 98%. Cabeceira do leito "ZERO GRA... |
| 9095 | S/ | SUCESSO | 1 | no_relation | (nenhum) | 1 | ...O PARA PCR. REALIZADO MANOBRAS DE REANIMAÇÃO S/ SUCESSO, EVOLUIU PARA ÓBITO AS 0:30HS. FRATURA PERTR... |
| 9112 | evacuação | ausente | 1 | no_relation | (nenhum) | 1 | ...A +. PAM em radial D permeável. SVD anúrica, evacuação ausente. Edema e exudato em MMSS, importante fragili... |
| 9112 | ausente | evacuação | 1 | no_relation | (nenhum) | 1 | ...A +. PAM em radial D permeável. SVD anúrica, evacuação ausente. Edema e exudato em MMSS, importante fragili... |
| 9118 | evacuação | ausente | 1 | no_relation | (nenhum) | 1 | ...so, flácido, RHA +, SVD com diurese efetiva, evacuação ausente, extremidades bem perfundidas, higienizado e... |
| 9215 | sem | queixas algicas | 1 | no_relation | (nenhum) | 1 | ...emodinamicamente, refere bom sono e repouso, sem queixas algicas, aceitando a dieta oferecida. Pele integra e... |
| 9215 | SEM | QUEIXAS | 1 | no_relation | (nenhum) | 1 | ...XXXX. EXAME GERAL E INVESTIGACAO DE PESSOAS SEM QUEIXAS OU DIAGNOSTICO RELATADO |
| 9298 | ausente | evacuação | 1 | no_relation | (nenhum) | 1 | ...eg umbilical. nega algia. diurese presente e evacuação ausente após cirurgia. corado, abdome flácido. avp p... |
| 9325 | SEM | LESÕES OBSTRUTIVAS SIGNIFICATIVAS | 1 | no_relation | (nenhum) | 1 | ...ISMO CARDÍACO , O QUAL DEMONSTROU CORONÁRIAS SEM LESÕES OBSTRUTIVAS SIGNIFICATIVAS , E ECOCARDIOGRAMA COM AE = 50 , VE = 56/37... |
| 9348 | Nega | palpitação | 1 | no_relation | (nenhum) | 1 | ...sforço. Discreto edema mmii pricn a esquerda.Nega palpitação.Dor a movimentação do torax. MV normodist s... |
| 9454 | NEGA | ALTERAÇÃO | 1 | no_relation | (nenhum) | 1 | ...TA , ,PAROU HÁ 6 MESES (FUMAVA DESDE 9 ANOS) NEGA ALTERAÇÃO TIRÓIDE, IAM, AVE EM USO DE LST 50 2X, AAS,... |
| 9540 | não | evidencia doença coronariana importante | 1 | no_relation | (nenhum) | 1 | ...ao sistolica pulmonar de 20mmHg. Cateterismo não evidencia doença coronariana importante Laboratorio 13/08/2014 Creatinina 0,98 Glic... |
| 9593 | SEM | PRESENCA DE SINTOMAS | 1 | no_relation | (nenhum) | 1 | ...SODIOS NAO SUSTENTADOS DE TAQUICARDIA ATRIAL SEM PRESENCA DE SINTOMAS. # PACIENTE REFERE ESTAR SE SENTINDO BEM COM... |
| 9844 | sem | alterações | 1 | no_relation | (nenhum) | 1 | ...reito, já presente em radiografias prévias e sem alterações; como a característica e a evolução radiográ... |
| 9861 | NÃO | FALTAR CONSULTA | 1 | no_relation | (nenhum) | 1 | ...- 50, mantido demais, retorno em 30 DIAS + NÃO FALTAR CONSULTA COM Doutor Vital Brazil. |

## Limite desta auditoria

O criterio decide por evidencia estrutural do gold, nao por leitura clinica caso
a caso. Ele acerta o padrao dominante, que e a lista coordenada, e vai errar em
casos que exigem saber o que a frase quer dizer. Uma revisao humana das linhas
acima pode mover casos entre as tres classes, e as colunas `trecho` e `alvos
anotados da pista` estao no relatorio exatamente para permitir isso.

