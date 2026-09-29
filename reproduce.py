"""在可信环境运行：重新训练，核对开发集预测和测试集答案，不需要测试参考标签。"""
import json
from pathlib import Path
import sms_lab as lab
root=Path(__file__).resolve().parent
s=json.loads((root/'结果.json').read_text(encoding='utf-8'))
saved=json.loads((root/'逐条预测.json').read_text(encoding='utf-8'))
source=(root/'student_model.py').read_text(encoding='utf-8')
assert lab.digest(source.encode())==s['plan']['source_sha256'],'代码摘要不符'
for name,sha in s['plan']['data_hashes'].items():assert lab.digest((root/'data'/name).read_bytes())==sha,'数据摘要不符'
model=lab.load_model(source,s['plan']['method'],s['plan']['config'])
dev=lab.read('dev.jsonl')
actual={'dev_baseline':lab.evaluate(dev),'dev_candidate':lab.evaluate(dev,model),'test_candidate':lab.evaluate(lab.read('test_texts.jsonl'),model)}
assert actual==saved,'逐条预测不一致，请检查版本、代码和随机种子'
assert {k:lab.counts(v) for k,v in actual.items() if k.startswith('dev_')}==s['counts'],'开发集计数不符'
assert (root/'test_predictions.csv').read_text(encoding='utf-8')==lab.answers_csv(actual['test_candidate']),'测试答案CSV不一致'
print('复现通过：开发集结果和543条测试集答案与提交一致。')
