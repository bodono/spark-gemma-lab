// A small paired A/B workload using the same runner and saved evidence schema.
// Keep the app idle while running this direct CLI (it bypasses the bridge lock).
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { validate, runExperiment } from '../server/runner.mjs';
const label = process.argv[2];
if (!label || !/^[a-z0-9-]+$/.test(label)) throw Error('Supply a lowercase tuning label');
const config = JSON.parse(await readFile('config/models.json', 'utf8'));
config.runtime.tuning_label = label;
const only = process.argv[3];
if (only && !['diffusion','autoregressive'].includes(only)) throw Error('Optional model must be diffusion or autoregressive');
if (only) config.models = config.models.filter(m => m.id === only);
const dataset = (await readFile('data/pg19-512.jsonl', 'utf8')).trim().split('\n').map(JSON.parse);
const settings = validate({kind:'profile',prompt:'Continue the passage.',max_tokens:512,batch_size:1,batch_sizes:[1,2,4],requests_per_condition:4,repeats:1,warmups:1,seed:42,temperature:0,dataset,dataset_name:'PG19 chat continuation · kernel A/B',workload:'continuation',output_mode:'natural'});
const controller = new AbortController();
process.on('SIGINT', () => controller.abort());
const run = await runExperiment(settings,config,{signal:controller.signal,emit(e){if(e.type==='phase')console.log(e.message);if(e.type==='summary')console.log(JSON.stringify(e.summary));}});
await mkdir('results/tuning',{recursive:true});
await writeFile(`results/tuning/${label}.json`,JSON.stringify(run,null,2));
console.log(`Saved results/tuning/${label}.json (${run.status})`);
if(run.status!=='complete')process.exitCode=2;
