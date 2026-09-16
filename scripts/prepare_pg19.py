#!/usr/bin/env python3
"""Prepare reproducible PG19 excerpts without downloading the training corpus."""
import argparse,base64,hashlib,json,random,urllib.parse,urllib.request
from pathlib import Path
INVENTORY_REVISION='4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b'
TOKENIZER='RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic'
TOKENIZER_REVISION='ed35d7abe5d940da41b4ff06eb482feb0be8cb44'
def fetch(url):
    with urllib.request.urlopen(url,timeout=120) as response:return response.read()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--count',type=int,default=96);p.add_argument('--input-tokens',type=int,default=512);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,default=Path('data/pg19-512.jsonl'));p.add_argument('--cache',type=Path,default=Path('data/pg19-cache'));p.add_argument('--tokenizer',default=TOKENIZER);p.add_argument('--tokenizer-revision',default=TOKENIZER_REVISION)
    a=p.parse_args()
    if not 1<=a.count<=20000 or not 1<=a.input_tokens<=16000:p.error('Invalid count or input token length')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,revision=a.tokenizer_revision)
    inventory_url=f'https://huggingface.co/datasets/deepmind/pg19/resolve/{INVENTORY_REVISION}/data/{a.split}_files.txt'
    names=sorted(fetch(inventory_url).decode().splitlines())
    listing_url='https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?'+urllib.parse.urlencode({'prefix':a.split+'/','maxResults':200})
    listing=json.loads(fetch(listing_url));objects={o['name']:o for o in listing['items']}
    if listing.get('nextPageToken') or set(names)!=set(objects):raise RuntimeError('PG19 inventory mismatch; inspect upstream changes')
    a.cache.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(a.seed);rng.shuffle(names);books={};rows=[];seen=set()
    for i in range(a.count):
        name=names[i%len(names)];obj=objects[name]
        url='https://storage.googleapis.com/deepmind-gutenberg/'+urllib.parse.quote(name,safe='/')+'?generation='+obj['generation']
        if name not in books:
            cachefile=a.cache/(name.replace('/','_')+'.'+obj['generation'])
            raw=cachefile.read_bytes() if cachefile.exists() else fetch(url)
            if len(raw)!=int(obj['size']) or base64.b64encode(hashlib.md5(raw).digest()).decode()!=obj['md5Hash']:raise RuntimeError('Checksum mismatch: '+name)
            if not cachefile.exists():cachefile.write_bytes(raw)
            ids=tokenizer.encode(raw.decode('utf-8'),add_special_tokens=False)
            if len(ids)<a.input_tokens:raise RuntimeError('Book too short: '+name)
            books[name]=(ids,hashlib.sha256(raw).hexdigest())
        ids,source_hash=books[name]
        for attempt in range(200):
            offset=rng.randrange(len(ids)-a.input_tokens+1)
            prompt=tokenizer.decode(ids[offset:offset+a.input_tokens],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            actual=len(tokenizer.encode(prompt,add_special_tokens=False));digest=hashlib.sha256(prompt.encode()).hexdigest()
            if actual==a.input_tokens and digest not in seen:break
        else:raise RuntimeError('Cannot produce a unique exact-length prompt; choose another seed')
        seen.add(digest)
        rows.append({'prompt':prompt,'sample_id':i,'corpus':'PG19','split':a.split,'book_id':name.rsplit('/',1)[1].removesuffix('.txt'),'token_offset':offset,'input_tokens':actual,'requested_input_tokens':a.input_tokens,'seed':a.seed,'tokenizer':a.tokenizer,'tokenizer_revision':a.tokenizer_revision,'prompt_sha256':digest,'source_url':url,'source_generation':obj['generation'],'source_sha256':source_hash})
        print(f'Prepared {i+1}/{a.count}',flush=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    a.output.with_suffix('.manifest.json').write_text(json.dumps({'inventory_url':inventory_url,'inventory_revision':INVENTORY_REVISION,'object_inventory':listing,'seed':a.seed,'tokenizer':a.tokenizer,'tokenizer_revision':a.tokenizer_revision,'count':a.count,'input_tokens':a.input_tokens,'prompts_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
    print(a.output.resolve())
if __name__=='__main__':main()
