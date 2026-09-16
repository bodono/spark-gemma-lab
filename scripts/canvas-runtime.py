#!/usr/bin/env python3
"""Owned Spark Lab canvas controller. JSON stdout; human progress stderr.

Usage: python3 canvas-runtime.py status
       python3 canvas-runtime.py ensure 8|16|32|64|128|256|512

No GPU work. A detached worker owns each locked stop/start transition, allowing
an interrupted SSH client to reconnect safely. Only validated API/EngineCore
identities may receive SIGTERM; SIGKILL and process-group signals are never used.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

ALLOWED = (8, 16, 32, 64, 128, 256, 512)
CHECKPOINT = 'RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic'
REVISION = '3b3dae4697494da5a290e9c0461954449e76c4f5'
SERVED = 'diffusiongemma-fp8'
PORT = 8000
TRANSITION_SECONDS = 480
TERMINAL = {'complete', 'failed'}


class Refusal(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def progress(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except (BrokenPipeError, OSError):
        pass


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2)+'\n').encode())


def atomic_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def option(args, flag):
    values = []
    for i, arg in enumerate(args):
        if arg == flag:
            if i+1 >= len(args):
                raise Refusal('invalid_command', f'Missing value for {flag}')
            values.append(args[i+1])
        elif arg.startswith(flag+'='):
            values.append(arg.split('=', 1)[1])
    if len(values) != 1:
        raise Refusal('invalid_command', f'Expected exactly one {flag}')
    return values[0]


class Runtime:
    def __init__(self, root=None, proc=None, uid=None):
        self.root = Path(root) if root is not None else Path.home()/'.local/share/spark-gemma-lab'
        self.proc = Path(proc) if proc is not None else Path('/proc')
        self.uid = os.geteuid() if uid is None else uid
        self.venv_python = self.root/'venv/bin/python'
        self.vllm = self.root/'venv/bin/vllm'
        self.launcher = self.root/'serve-spark-user.sh'
        self.pidfile = self.root/'diffusion.pid'
        self.lockfile = self.root/'canvas-controller.lock'
        self.statefile = self.root/'canvas-controller-state.json'
        self.server_log = self.root/'logs/diffusion.log'
        self.controller = Path(__file__).resolve()
        self.base_url = f'http://127.0.0.1:{PORT}'

    def boot_id(self):
        try:
            return (self.proc/'sys/kernel/random/boot_id').read_text().strip()
        except OSError as exc:
            raise Refusal('proc_unavailable', 'Cannot read Linux boot identity') from exc

    def process(self, pid):
        if type(pid) is not int or pid <= 1:
            return None
        path = self.proc/str(pid)
        try:
            stat = (path/'stat').read_text()
            fields = stat[stat.rfind(')')+2:].split()
            status = (path/'status').read_text()
            uids = next(l for l in status.splitlines() if l.startswith('Uid:')).split()[1:]
            args = [a for a in (path/'cmdline').read_bytes().decode().split('\0') if a]
            return {'pid':pid, 'ppid':int(fields[1]), 'starttime':fields[19],
                    'state':fields[0], 'uid':path.stat().st_uid,
                    'uids':[int(u) for u in uids], 'args':args,
                    'exe':str((path/'exe').resolve(strict=True)), 'boot_id':self.boot_id()}
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, ValueError, StopIteration, UnicodeDecodeError, IndexError) as exc:
            raise Refusal('proc_unreadable', f'Cannot verify process {pid}') from exc

    def same(self, first, second):
        return bool(first and second and all(first[k] == second[k]
                   for k in ('pid','starttime','uid','boot_id')))

    def own_uid(self, process):
        return process['uid'] == self.uid and process['uids'] == [self.uid]*4

    def api(self, process):
        if not process or process['state'] == 'Z':
            return None
        args = process['args']
        if not self.own_uid(process):
            raise Refusal('foreign_uid', f'Process {process["pid"]} is not owned by this user')
        if (len(args) < 4 or args[:4] != [str(self.venv_python), str(self.vllm), 'serve', CHECKPOINT]
                or process['exe'] != str(self.venv_python.resolve(strict=True))):
            raise Refusal('unmanaged_process', f'Process {process["pid"]} is not the exact owned vLLM server')
        expected = {'--revision':REVISION, '--served-model-name':SERVED,
                    '--host':'127.0.0.1', '--port':str(PORT)}
        for key, value in expected.items():
            if option(args, key) != value:
                raise Refusal('unmanaged_process', f'Owned-server identity mismatch: {key}')
        try:
            diffusion = json.loads(option(args, '--diffusion-config'))
            canvas = diffusion['canvas_length']
            if type(canvas) is not int or canvas not in ALLOWED:
                raise ValueError('canvas')
            config = {
                'canvas_length':canvas,
                'max_denoising_steps':int(diffusion['max_denoising_steps']),
                'max_model_len':int(option(args, '--max-model-len')),
                'max_num_seqs':int(option(args, '--max-num-seqs')),
                'max_num_batched_tokens':int(option(args, '--max-num-batched-tokens')),
                'attention_backend':option(args, '--attention-backend'),
            }
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise Refusal('invalid_configuration', 'Cannot attest owned runtime configuration') from exc
        if config['max_denoising_steps'] != 48 or config['attention_backend'] != 'TRITON_ATTN':
            raise Refusal('unexpected_configuration', 'Expected adaptive cap48 and TRITON_ATTN')
        if config['max_model_len'] != 8192 or config['max_num_batched_tokens'] != 8192:
            raise Refusal('unexpected_configuration', 'Expected 8192 context and token budget; refusing to silently change them')
        if any(a == '--compilation-config' or a.startswith('--compilation-config=') for a in args):
            raise Refusal('unexpected_configuration', 'Custom graph configuration requires an explicit controller update')
        if not 1 <= config['max_num_seqs'] <= 128:
            raise Refusal('unexpected_configuration', 'Invalid owned capacity')
        return {**process, 'configuration':config,
                'config_hash':hashlib.sha256(json.dumps(args[2:],separators=(',',':')).encode()).hexdigest(),
                'canvas_length':canvas}

    def all_processes(self):
        result = {}
        for path in self.proc.iterdir():
            if not path.name.isdigit():
                continue
            # Other users' processes are irrelevant for API discovery/ancestry.
            try:
                if path.stat().st_uid != self.uid:
                    continue
                p = self.process(int(path.name))
            except (Refusal,OSError):
                continue
            if p:
                result[p['pid']] = p
        return result

    def listeners(self):
        result = []
        for family in ('tcp','tcp6'):
            try:
                lines = (self.proc/'net'/family).read_text().splitlines()[1:]
            except FileNotFoundError:
                if family == 'tcp6':
                    continue
                raise Refusal('proc_unavailable', 'Cannot inspect listening TCP sockets')
            for line in lines:
                cols = line.split()
                if int(cols[1].split(':')[1],16) == PORT and cols[3] == '0A':
                    result.append({'family':family,'address':cols[1].split(':')[0],
                                   'uid':int(cols[7]),'inode':cols[9]})
        return result

    def check_listener(self, api):
        listeners = self.listeners()
        if not listeners:
            return False
        if api is None:
            raise Refusal('unmanaged_port', f'Port {PORT} is occupied without a verified owned API')
        try:
            sockets = set()
            for path in (self.proc/str(api['pid'])/'fd').iterdir():
                try:
                    sockets.add(os.readlink(path))
                except FileNotFoundError:
                    continue  # An unrelated descriptor may close during this scan.
        except OSError as exc:
            raise Refusal('socket_owner_unknown', 'Cannot verify listener ownership') from exc
        for listener in listeners:
            if (listener['uid'] != self.uid or listener['family'] != 'tcp'
                    or listener['address'] != '0100007F'
                    or f'socket:[{listener["inode"]}]' not in sockets):
                raise Refusal('unmanaged_port', f'Port {PORT} has an unmanaged listener')
        return True

    def find_api(self):
        if self.pidfile.exists():
            try:
                pid = int(self.pidfile.read_text().strip())
            except (OSError, ValueError) as exc:
                raise Refusal('invalid_pidfile', 'Cannot verify diffusion.pid') from exc
            current = self.process(pid)
            if current and current['state'] != 'Z':
                return self.api(current)  # A reused/non-owned PID is a hard refusal.
        found = []
        for p in self.all_processes().values():
            if len(p['args']) > 1 and p['args'][1] == str(self.vllm):
                found.append(self.api(p))
        found = [p for p in found if p]
        if len(found) > 1:
            raise Refusal('ambiguous_runtime', 'More than one owned API server exists')
        return found[0] if found else None

    def http(self, path, timeout=3):
        with urllib.request.urlopen(self.base_url+path, timeout=timeout) as response:
            return response.read().decode()

    def health(self):
        try:
            self.http('/health',2)
            return True
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def idle_metrics(self):
        try:
            data = self.http('/metrics',3)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise Refusal('idle_unverified', 'Cannot prove runtime idle: metrics unavailable') from exc
        values = {'running':[], 'waiting':[]}
        pattern = re.compile(r'^vllm:num_requests_(running|waiting)(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?$')
        for line in data.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            labels = match.group(2) or ''
            model = re.search(r'(?:[{,])model_name="([^"\\]*)"(?:[,}])',labels)
            if not model or model.group(1) != SERVED:
                continue
            try:
                number = float(match.group(3))
            except ValueError as exc:
                raise Refusal('idle_unverified', 'Invalid active-request metric') from exc
            if not math.isfinite(number) or number < 0:
                raise Refusal('idle_unverified', 'Invalid active-request metric')
            values[match.group(1)].append(number)
        if not all(values.values()):
            raise Refusal('idle_unverified', 'Both running and waiting request gauges are required')
        result = {name:sum(rows) for name,rows in values.items()}
        if result['running'] or result['waiting']:
            raise Refusal('runtime_busy', f'Refusing restart: {result["running"]:g} running, {result["waiting"]:g} waiting requests')
        return result

    def attestation(self, api, ready, changed=False):
        return {'ok':True, 'running':bool(api), 'ready':bool(ready),
                'verified':bool(api), 'changed':bool(changed),
                'canvas_length':api['canvas_length'] if api else None,
                'pid':api['pid'] if api else None,
                'starttime':api['starttime'] if api else None,
                'uid':api['uid'] if api else None,
                'boot_id':api['boot_id'] if api else self.boot_id(),
                'config_hash':api['config_hash'] if api else None,
                'configuration':api['configuration'] if api else None}

    def status(self):
        state = read_json(self.statefile)
        try:
            api = self.find_api()
        except Refusal:
            # A launcher keeps the future API PID during patch preparation.
            spawn = state.get('spawn') if state and state.get('phase') not in TERMINAL else None
            p = self.process(spawn['pid']) if spawn else None
            if spawn and self.same(spawn,p) and self.own_uid(p) and p['args'] == ['/bin/bash',str(self.launcher),'diffusion']:
                result = self.attestation(None,False)
                result.update(transition=state['phase'],target_canvas=state['target_canvas'],operation_id=state['operation_id'])
                return result
            raise
        listening = self.check_listener(api)
        result = self.attestation(api,listening and self.health())
        if state and state.get('phase') not in TERMINAL:
            result.update(transition=state['phase'],target_canvas=state['target_canvas'],operation_id=state['operation_id'])
        return result

    def validate_launcher(self):
        for path in (self.root,self.launcher,self.vllm,self.controller):
            if not path.exists() or path.stat().st_uid != self.uid:
                raise Refusal('unowned_files', f'Missing or non-owned runtime path: {path}')
            if path.stat().st_mode & 0o022:
                raise Refusal('unsafe_permissions', f'Group/world-writable runtime path: {path}')
        if 'CANVAS_LENGTH' not in self.launcher.read_text():
            raise Refusal('launcher_outdated', 'Launcher does not support CANVAS_LENGTH; refusing to stop server')
        if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
            raise Refusal('pidfd_required','Linux pidfd support is required for safe signaling')

    def descendants(self, api):
        table = self.all_processes()
        engines = []
        for p in table.values():
            if not self.own_uid(p) or p['exe'] != api['exe'] or not p['args']:
                continue
            if not re.fullmatch(r'VLLM::EngineCore(?:_DP\d+)?',p['args'][0]) or len(p['args']) != 1:
                continue
            ancestor = p['ppid']
            seen = set()
            while ancestor in table and ancestor not in seen:
                if ancestor == api['pid']:
                    engines.append(p)
                    break
                seen.add(ancestor)
                ancestor = table[ancestor]['ppid']
        return engines

    def live_identity(self, expected):
        p = self.process(expected['pid'])
        return p if self.same(expected,p) and p['state'] != 'Z' else None

    def terminate(self, expected, engine=False):
        # pidfd binds the signal to a process instance, not a recyclable PID.
        try:
            fd = os.pidfd_open(expected['pid'])
        except ProcessLookupError:
            return
        try:
            p = self.live_identity(expected)
            if p is None:
                return
            if not self.own_uid(p) or p['exe'] != expected['exe'] or p['args'] != expected['args']:
                raise Refusal('identity_changed','Refusing signal after process identity changed')
            if engine:
                if not re.fullmatch(r'VLLM::EngineCore(?:_DP\d+)?',p['args'][0]) or len(p['args']) != 1:
                    raise Refusal('unmanaged_engine','Refusing signal to non-EngineCore process')
            else:
                self.api(p)
            if p['state'] in ('T','t'):
                raise Refusal('paused_process','Refusing signal to a suspended process')
            signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
        finally:
            os.close(fd)

    def wait_gone(self, identities, deadline):
        while time.monotonic() < deadline:
            if not any(self.live_identity(p) for p in identities):
                return True
            time.sleep(.25)
        return not any(self.live_identity(p) for p in identities)

    def state(self, operation_id, target, phase, **extra):
        previous = read_json(self.statefile) or {}
        if previous.get('operation_id') not in (None,operation_id):
            raise Refusal('operation_changed','Controller operation identity changed')
        record = {**previous,'operation_id':operation_id,'target_canvas':target,
                  'phase':phase,'updated_at':time.time(),**extra}
        atomic_json(self.statefile,record)
        progress(f'Canvas {target}: {phase}')
        return record

    def transition(self, target, operation_id):
        deadline = time.monotonic()+TRANSITION_SECONDS
        self.validate_launcher()
        api = self.find_api()
        self.check_listener(api)
        if api and api['canvas_length'] == target:
            self.state(operation_id,target,'waiting_ready')
            result = self.wait_ready(api,target,deadline,changed=False)
            self.idle_metrics()
            return result
        engines = self.descendants(api) if api else []
        if api:
            self.state(operation_id,target,'checking_idle',previous_api=api,previous_engines=engines)
            if not self.check_listener(api) or not self.health():
                raise Refusal('runtime_not_ready','Existing API is not ready; cannot prove it idle')
            self.idle_metrics()
            time.sleep(.15)
            self.idle_metrics()
            self.check_listener(self.api(self.process(api['pid'])))
            if not self.same(api,self.process(api['pid'])):
                raise Refusal('identity_changed','API changed during idle verification')
            self.state(operation_id,target,'stopping')
            self.terminate(api)
            targets = [api,*engines]
            if not self.wait_gone(targets,min(deadline,time.monotonic()+25)):
                self.state(operation_id,target,'stopping_engines')
                for engine in engines:
                    if self.live_identity(engine):
                        self.terminate(engine,engine=True)
                if not self.wait_gone(targets,min(deadline,time.monotonic()+30)):
                    raise Refusal('shutdown_timeout','Owned API/EngineCore did not exit after SIGTERM; no force kill attempted')
        else:
            # Recover only EngineCore identities saved while verified descendants.
            previous = read_json(self.statefile) or {}
            for engine in previous.get('previous_engines',[]):
                if self.live_identity(engine):
                    raise Refusal('orphaned_owned_engine','A recorded prior EngineCore still runs; inspect it before restart')
        self.check_listener(None)
        if time.monotonic() >= deadline:
            raise Refusal('transition_timeout','Canvas transition deadline reached before launch')
        config = api['configuration'] if api else {'max_denoising_steps':48,'max_num_seqs':4,'attention_backend':'TRITON_ATTN'}
        env = dict(os.environ)
        env.update(CANVAS_LENGTH=str(target),ENABLE_TORCH_PROFILER='1',
                   MAX_DENOISING_STEPS=str(config['max_denoising_steps']),
                   MAX_NUM_SEQS=str(config['max_num_seqs']),ATTENTION_BACKEND=config['attention_backend'],
                   EXTEND_DIFFUSION_GRAPHS='0',USE_TUNED_MOE='0',ALLOW_BUSY='0')
        self.server_log.parent.mkdir(parents=True,exist_ok=True)
        with self.server_log.open('ab',buffering=0) as log:
            child = subprocess.Popen(['/bin/bash',str(self.launcher),'diffusion'],
                       stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,
                       cwd=self.root,start_new_session=True,close_fds=True)
        spawn = None
        for _ in range(20):
            spawn = self.process(child.pid)
            if spawn:
                break
            time.sleep(.05)
        if not spawn:
            raise Refusal('launch_failed','Owned launcher exited before process identity could be recorded')
        self.state(operation_id,target,'starting',spawn=spawn,log_path=str(self.server_log))
        atomic_bytes(self.pidfile,f'{child.pid}\n'.encode())
        return self.wait_ready(spawn,target,deadline,changed=True,child=child)

    def wait_ready(self, identity, target, deadline, changed, child=None):
        while time.monotonic() < deadline:
            p = self.live_identity(identity)
            if p is None:
                raise Refusal('startup_exited',f'Owned server exited; inspect {self.server_log}')
            # Launcher performs existing guarded patches before exec'ing vLLM.
            if p['args'] == ['/bin/bash',str(self.launcher),'diffusion'] and changed:
                time.sleep(.5)
                continue
            api = self.api(p)
            if api['canvas_length'] != target:
                raise Refusal('canvas_mismatch',f'Actual canvas {api["canvas_length"]} differs from requested {target}')
            listening = self.check_listener(api)
            if listening and self.health():
                # Recheck identity after health to avoid attesting a changed PID.
                if not self.same(api,self.process(api['pid'])):
                    raise Refusal('identity_changed','API identity changed during readiness check')
                atomic_bytes(self.pidfile,f'{api["pid"]}\n'.encode())
                return self.attestation(api,True,changed)
            time.sleep(1)
        raise Refusal('readiness_timeout',f'Server was not ready within {TRANSITION_SECONDS}s; it may still be loading; inspect {self.server_log}')

    def ensure(self, target):
        self.root.mkdir(parents=True,exist_ok=True)
        fd = os.open(self.lockfile,os.O_RDWR|os.O_CREAT,0o600)
        try:
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                state = read_json(self.statefile) or {}
                if state.get('target_canvas') != target:
                    raise Refusal('transition_busy','A different canvas transition is in progress')
                operation_id = state.get('operation_id')
                if not operation_id:
                    raise Refusal('transition_busy','A canvas transition is starting; retry status')
            else:
                status = self.status()
                if status['ready'] and status['verified'] and status['canvas_length'] == target:
                    self.idle_metrics()
                    again = self.status()
                    if any(status[k] != again[k] for k in ('pid','starttime','config_hash')) or not again['ready']:
                        raise Refusal('identity_changed','Runtime changed while checking idle status')
                    return again
                previous = read_json(self.statefile) or {}
                operation_id = str(uuid.uuid4())
                atomic_json(self.statefile,{'operation_id':operation_id,'target_canvas':target,
                            'phase':'queued','started_at':time.time(),
                            'previous_engines':previous.get('previous_engines',[])})
                logpath = self.root/'logs'/f'canvas-controller-{operation_id}.log'
                logpath.parent.mkdir(parents=True,exist_ok=True)
                with logpath.open('ab',buffering=0) as log:
                    subprocess.Popen([sys.executable,str(self.controller),'_worker',str(target),operation_id,str(fd)],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=self.root,
                        start_new_session=True,close_fds=True,pass_fds=(fd,))
                # Do not LOCK_UN: the child inherited the same locked open-file
                # description. Closing this parent's FD leaves its lock intact.
        finally:
            os.close(fd)
        return self.wait_operation(operation_id,target)

    def wait_operation(self, operation_id, target):
        deadline = time.monotonic()+TRANSITION_SECONDS+15
        last_phase = None
        while time.monotonic() < deadline:
            state = read_json(self.statefile) or {}
            if state.get('operation_id') != operation_id:
                raise Refusal('operation_changed','Canvas operation was replaced; inspect status')
            phase = state.get('phase')
            if phase != last_phase:
                progress(f'Canvas {target}: {phase}')
                last_phase = phase
            if phase == 'complete':
                return state['result']
            if phase == 'failed':
                raise Refusal(state['error']['code'],state['error']['message'])
            time.sleep(.5)
        raise Refusal('controller_timeout','Transition continues independently; reconnect with status or ensure')


def worker(runtime, target, operation_id, fd):
    # The parent passes its held lock, atomically handing off ownership.
    lock_stat = runtime.lockfile.stat()
    inherited_stat = os.fstat(fd)
    if (lock_stat.st_dev,lock_stat.st_ino)!=(inherited_stat.st_dev,inherited_stat.st_ino):
        raise Refusal('invalid_lock','Inherited lock does not belong to this runtime')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:
        runtime.state(operation_id,target,'validating',worker=runtime.process(os.getpid()))
        result = runtime.transition(target,operation_id)
        runtime.state(operation_id,target,'complete',result=result)
    except Exception as exc:
        code = exc.code if isinstance(exc,Refusal) else 'controller_error'
        runtime.state(operation_id,target,'failed',error={'code':code,'message':str(exc)})
    finally:
        os.close(fd)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    runtime = Runtime()
    if len(args)==4 and args[0]=='_worker':
        target=int(args[1])
        if target not in ALLOWED:
            raise Refusal('invalid_canvas','Allowed canvas lengths:8,16,32,64,128,256,512')
        worker(runtime,target,args[2],int(args[3]))
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','ensure'])
    parser.add_argument('canvas',nargs='?',type=int,choices=ALLOWED)
    parsed=parser.parse_args(args)
    if parsed.action=='ensure' and parsed.canvas is None:
        parser.error('ensure requires a canvas length')
    try:
        result=runtime.status() if parsed.action=='status' else runtime.ensure(parsed.canvas)
        print(json.dumps(result,sort_keys=True),flush=True)
        return 0
    except Exception as exc:
        code=exc.code if isinstance(exc,Refusal) else 'controller_error'
        print(json.dumps({'ok':False,'ready':False,'verified':False,'changed':False,
                          'error':{'code':code,'message':str(exc)}},sort_keys=True),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
