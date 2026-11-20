"""Curva precisão x recall x F1 de `negation_of` por limiar de P(neg), só DEV.

Regra: se P(neg) >= t, prevê `negation_of`; senão, argmax entre as outras duas
classes. `t=None` é o argmax. Intervalos: bootstrap por documento (10.000,
semente 42, percentil 95%) das diferenças em relação ao argmax da mesma execução.
Descritivo: não escolhe limiar. Não lê o TEST.

Uso: python diagnosticos/curva_limiar_negation_of.py [RAIZ_CODIGO] [SAIDA.json]
RAIZ_CODIGO deve conter dados/particoes-v2/dev.jsonl e resultados/execucoes/.
"""
import json, sys, hashlib, numpy as np
from pathlib import Path
R=(sys.argv[1] if len(sys.argv)>1 else '.').rstrip('/')+'/'
SAIDA=sys.argv[2] if len(sys.argv)>2 else 'resultados/diagnosticos/curva_limiar_negation_of.json'
sys.path.insert(0,R)
from reclin.tarefa import candidatos
docs=[json.loads(l) for l in open(R+'dados/particoes-v2/dev.jsonl')]
C=candidatos(docs); doc_ids=np.array([c['doc_id'] for c in C])
ud,inv=np.unique(doc_ids,return_inverse=True)
def carregar(n):
    d=json.load(open(R+f'resultados/execucoes/{n}/predicoes_dev.json'))
    return np.array(d['y_true']),np.array(d['y_pred']),np.array(d['probs'])
def pred(p,t):
    if t is None: return p.argmax(1)
    out=p[:,1:].argmax(1)+1
    out[p[:,0]>=t]=0
    return out
def contagens(y,pr):
    pos=(pr==0)
    return (pos&(y==0)).sum(),(pos&(y!=0)).sum(),((~pos)&(y==0)).sum()
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan
    f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    return p,r,f
out={}
for nome,rot in [('pair_aware__a3ac671a77e2__t2','sem_pesos'),('pairaware_biobertpt_seed42','ponderada')]:
    y,yp,P=carregar(nome)
    assert len(y)==len(C)
    from reclin.tarefa import y_true
    assert (np.array(y_true(C))==y).all(), 'y_true diverge'
    assert (yp==P.argmax(1)).mean()>0.999, (yp==P.argmax(1)).mean()
    ts=[None]+[round(x,2) for x in np.arange(0.5,0.96,0.05)]+[0.97,0.99]
    # contagens por documento
    rng=np.random.default_rng(42)
    B=10000; idx=rng.integers(0,len(ud),(B,len(ud)))
    def por_doc(pr):
        pos=pr==0
        tp=np.bincount(inv,weights=(pos&(y==0)),minlength=len(ud))
        fp=np.bincount(inv,weights=(pos&(y!=0)),minlength=len(ud))
        fn=np.bincount(inv,weights=((~pos)&(y==0)),minlength=len(ud))
        return np.stack([tp,fp,fn])
    base=por_doc(pred(P,None))
    def boot(m):
        TP,FP,FN=[m[i][idx].sum(1) for i in range(3)]
        with np.errstate(all='ignore'):
            return TP/(TP+FP),TP/(TP+FN),2*TP/(2*TP+FP+FN)
    bb=boot(base)
    linhas=[]
    for t in ts:
        pr=pred(P,t); tp,fp,fn=contagens(y,pr); p,r,f=prf(tp,fp,fn)
        m=por_doc(pr); b=boot(m)
        ic=lambda a:[float(np.nanpercentile(a,2.5)),float(np.nanpercentile(a,97.5))]
        d=[ic(b[i]-bb[i]) for i in range(3)]
        linhas.append(dict(t=t,tp=int(tp),fp=int(fp),fn=int(fn),precisao=p,recall=r,f1=f,
          ic_prec=ic(b[0]),ic_rec=ic(b[1]),ic_f1=ic(b[2]),d_prec=d[0],d_rec=d[1],d_f1=d[2]))
    out[rot]=linhas
    print(rot)
    for l in linhas:
        print(f"t={l['t']}  VP{l['tp']:4d} FP{l['fp']:4d} FN{l['fn']:3d}  P={l['precisao']:.3f} R={l['recall']:.3f} F1={l['f1']:.3f}  ΔF1[{l['d_f1'][0]:+.3f};{l['d_f1'][1]:+.3f}] ΔP[{l['d_prec'][0]:+.3f};{l['d_prec'][1]:+.3f}] ΔR[{l['d_rec'][0]:+.3f};{l['d_rec'][1]:+.3f}]")
shas={n:hashlib.sha256(open(R+f'resultados/execucoes/{n}/predicoes_dev.json','rb').read()).hexdigest() for n in ('pair_aware__a3ac671a77e2__t2','pairaware_biobertpt_seed42')}
meta=dict(descricao="negation_of: P/R/F1 por limiar de P(neg), DEV, bootstrap por documento das diferenças ao argmax",
  regra="P(neg)>=t prevê negation_of; senão argmax entre associated_with e no_relation",
  n_boot=10000,seed=42,n_documentos=int(len(ud)),n_candidatos=len(C),probs_arredondadas_4_casas=True,
  predicoes_sha256=shas,test_lido=False,execucoes={'sem_pesos':'pair_aware__a3ac671a77e2__t2','ponderada':'pairaware_biobertpt_seed42'})
out={'meta':meta,**out}
Path(SAIDA).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(SAIDA,'w'),indent=1,ensure_ascii=False)
