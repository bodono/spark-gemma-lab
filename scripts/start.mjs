import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const cwd = fileURLToPath(new URL('../', import.meta.url));
const children = [
  spawn(process.execPath, ['server/index.mjs'], { cwd, stdio: 'inherit' }),
  spawn(
    process.execPath,
    [
      'node_modules/vinext/dist/cli.js',
      process.argv.includes('--production') ? 'start' : 'dev',
      '--hostname',
      '127.0.0.1',
    ],
    {
      cwd,
      stdio: 'inherit',
    },
  ),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) child.kill('SIGTERM');
  process.exitCode = code;
}
for (const child of children) child.on('exit', (code) => stop(code || 0));
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
