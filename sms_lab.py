"""W02-SMS-3.0 统一数据、比较与提交工具。分类器实现位于 student_model.py。"""
from pathlib import Path
import json,re,hashlib,types,zipfile,platform,importlib.metadata,csv,io
ROOT=Path(__file__).resolve().parent
VERSION='W02-SMS-3.0'
BASELINE={'keywords':['free','win','prize','cash'],'min_hits':1}
def read(name):return [json.loads(x) for x in (ROOT/'data'/name).read_text(encoding='utf-8').splitlines() if x.strip()]
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(b):return hashlib.sha256(b).hexdigest()
def load_model(source,method,config):
    module=types.ModuleType('student_model')
    exec(compile(source,'student_model.py','exec'),module.__dict__)
    train=read('train.jsonl')
    return module.build_classifier([r['text'] for r in train],[r['label'] for r in train],method,config)
def baseline_predictions(texts):
    return ['spam' if set(re.findall('[a-z]+',t.lower())) & set(BASELINE['keywords']) else 'ham' for t in texts]
def evaluate(rows,model=None):
    predictions=list(model.predict([r['text'] for r in rows])) if model is not None else baseline_predictions([r['text'] for r in rows])
    if len(predictions)!=len(rows) or any(p not in ['ham','spam'] for p in predictions):raise ValueError('predict(texts) 必须按输入顺序、每条返回一个 ham 或 spam。')
    return [dict(r,prediction=str(p)) for r,p in zip(rows,predictions)]
def counts(rows):
    return {'正常判对':sum(r['label']=='ham' and r['prediction']=='ham' for r in rows),'误拦':sum(r['label']=='ham' and r['prediction']=='spam' for r in rows),'垃圾检出':sum(r['label']=='spam' and r['prediction']=='spam' for r in rows),'漏检':sum(r['label']=='spam' and r['prediction']=='ham' for r in rows)}
def show_data():
    from collections import Counter
    for filename in ['train.jsonl','dev.jsonl']:
        rows=read(filename);print(filename,len(rows),dict(Counter(r['label'] for r in rows)))
    for label in ['ham','spam']:
        row=next(r for r in read('dev.jsonl') if r['label']==label)
        print('\n',row['id'],row['label'],'\n',row['text'])
def baseline_dev():
    rows=evaluate(read('dev.jsonl'));print('规则基线：',counts(rows));return rows
def show_errors(rows,kind=None,limit=5):
    if rows is None:return
    selected=[r for r in rows if r['label']!=r['prediction'] and (kind is None or r['label']=={'误拦':'ham','漏检':'spam'}[kind])]
    print('共有',len(selected),'条错误，显示前',min(limit,len(selected)),'条。')
    for r in selected[:limit]:print('\n',r['id'],'参考:',r['label'],'预测:',r['prediction'],'\n',r['text'])
def fit_and_compare(method,config):
    source=(ROOT/'student_model.py').read_text(encoding='utf-8')
    config=json.loads(json.dumps(config))
    model=load_model(source,method,config)
    rows=read('dev.jsonl');base=evaluate(rows);candidate=evaluate(rows,model)
    print('开发集规则基线：',counts(base));print('开发集我的方法：',counts(candidate))
    return {'method':method,'config':config,'source':source,'model':model,'dev_baseline':base,'dev_candidate':candidate}
def signature(bundle,reason):
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    return {'version':VERSION,'method':bundle['method'],'config':bundle['config'],'source_sha256':digest(bundle['source'].encode()),'reason':reason.strip(),'data_hashes':manifest['files']}
def answers_csv(rows):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n');writer.writerow(['id','label'])
    writer.writerows((r['id'],r['prediction']) for r in rows)
    return stream.getvalue()
def predict_test(bundle,reason):
    if not reason.strip():print('先填写方法选择理由，再生成测试集答案。');return None
    # 测试文件只有 id/text，学生端没有读取测试参考答案的步骤。
    texts=read('test_texts.jsonl');predictions=evaluate(texts,bundle['model'])
    outputs={'dev_baseline':bundle['dev_baseline'],'dev_candidate':bundle['dev_candidate'],'test_candidate':predictions}
    formal={'plan':signature(bundle,reason),'source':bundle['source'],'outputs':outputs,'counts':{k:counts(v) for k,v in outputs.items() if k.startswith('dev_')}}
    write(ROOT/'personal/测试预测记录.json',formal)
    (ROOT/'personal/test_predictions.csv').write_text(answers_csv(predictions),encoding='utf-8')
    print(f'已生成 {len(predictions)} 条测试答案：personal/test_predictions.csv')
    print('列为 id,label，其中 label 是你的程序预测的 ham 或 spam。参考答案由教师保留。')
    print(answers_csv(predictions[:3]))
    return formal
def export_submission(student_id,notes,formal):
    if formal is None:print('先生成测试集答案。');return None
    try:
        if not isinstance(student_id,str) or not re.fullmatch('[A-Za-z0-9_-]+',student_id):raise ValueError('请填写学号。')
        for key in ['fp_id','fp_explanation','fn_id','fn_explanation','conclusion','ai_used']:
            if not isinstance(notes.get(key),str) or not notes[key].strip():raise ValueError('请填写：'+key)
        base={r['id']:r for r in formal['outputs']['dev_baseline']}
        for field,actual,pred in [('fp_id','ham','spam'),('fn_id','spam','ham')]:
            row=base.get(notes[field].strip())
            if not row or (row['label'],row['prediction'])!=(actual,pred):raise ValueError(field+' 应对应规则基线在开发集的这类错误。')
        if formal!=json.loads((ROOT/'personal/测试预测记录.json').read_text(encoding='utf-8')):raise ValueError('当前结果与最近生成的测试预测记录不一致，请重新运行生成与导出单元格。')
        versions={'python':platform.python_version()}
        for name in ['nltk','scikit-learn','numpy','scipy']:
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:pass
        summary={'version':VERSION,'student_id':student_id,'plan':formal['plan'],'counts':formal['counts'],'notes':notes,'runtime':versions}
        output=ROOT/'personal'/('W02_'+student_id+'.zip')
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('结果.json',json.dumps(summary,ensure_ascii=False,indent=2))
            z.writestr('逐条预测.json',json.dumps(formal['outputs'],ensure_ascii=False,indent=2))
            z.writestr('student_model.py',formal['source'])
            z.writestr('test_predictions.csv',answers_csv(formal['outputs']['test_candidate']))
            for name in ['sms_lab.py','reproduce.py','requirements.txt']:z.write(ROOT/name,name)
            for name in ['train.jsonl','dev.jsonl','test_texts.jsonl','manifest.json','README.md','原始数据说明.txt']:
                z.write(ROOT/'data'/name,'data/'+name)
            z.writestr('README.txt','本包含训练数据、模型构建代码、开发集结果与测试集预测答案 test_predictions.csv（id,label）。测试参考标签由教师保留，不在本包中。安装 requirements.txt 后运行 python reproduce.py 可重新训练并对照预测。自定义方法的附加依赖须写入 requirements.txt，所有实现放在 student_model.py。')
        print('已生成：',output,'\n请上传该 ZIP，并完成第二课 5 题。');return output
    except (ValueError,OSError) as e:print('尚未导出：',e);return None
