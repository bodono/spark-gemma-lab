#!/usr/bin/env python3
"""Safe CPU fixture tests: synthetic /proc, mocked HTTP/signals/spawns.

The only real child process verifies inherited flock lifetime in a temp folder;
it never imports the runtime controller or touches a real inference process.
"""
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('canvas_runtime',HERE.parent/'scripts/canvas-runtime.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Fixture:
    def __init__(self,folder):
        self.root=Path(folder)/'runtime';self.proc=Path(folder)/'proc'
        (self.root/'venv/bin').mkdir(parents=True)
        (self.root/'logs').mkdir()
        (self.root/'venv/bin/python').symlink_to(sys.executable)
        (self.root/'venv/bin/vllm').write_text('# owned executable\n')
        (self.root/'serve-spark-user.sh').write_text('#!/bin/bash\nCANVAS_LENGTH=${CANVAS_LENGTH:-256}\n')
        (self.proc/'sys/kernel/random').mkdir(parents=True)
        (self.proc/'sys/kernel/random/boot_id').write_text('fixture-boot')
        (self.proc/'net').mkdir()
        (self.proc/'net/tcp').write_text('header\n')
        (self.proc/'net/tcp6').write_text('header\n')
        self.runtime=m.Runtime(self.root,self.proc)
        self.runtime.controller=self.root/'canvas-runtime.py'
        self.runtime.controller.write_text('# controller fixture\n')
        self.uid=os.geteuid()
        self.metrics='\n'.join([f'vllm:num_requests_{name}{{engine="0",model_name="{m.SERVED}"}} 0.0' for name in ['running','waiting']])
        self.runtime.http=lambda path,timeout=3: '' if path=='/health' else self.metrics

    def args(self,canvas=256):
        return [str(self.root/'venv/bin/python'),str(self.root/'venv/bin/vllm'),'serve',m.CHECKPOINT,
                '--revision',m.REVISION,'--served-model-name',m.SERVED,'--host','127.0.0.1','--port','8000',
                '--diffusion-config',json.dumps({'canvas_length':canvas,'max_denoising_steps':48}),
                '--max-model-len','8192','--max-num-seqs','4','--max-num-batched-tokens','8192',
                '--attention-backend','TRITON_ATTN']

    def process(self,pid=70001,args=None,parent=1,start='100',state='S',uid=None):
        path=self.proc/str(pid);path.mkdir(exist_ok=True)
        fields=['0']*50;fields[0]=state;fields[1]=str(parent);fields[19]=str(start)
        (path/'stat').write_text(f'{pid} (fixture process) '+' '.join(fields))
        u=self.uid if uid is None else uid
        (path/'status').write_text('Name:\tfixture\nUid:\t'+ '\t'.join([str(u)]*4)+'\n')
        (path/'cmdline').write_bytes(('\0'.join(args if args is not None else self.args())+'\0').encode())
        if not (path/'exe').exists(): (path/'exe').symlink_to(Path(sys.executable).resolve())
        (path/'fd').mkdir(exist_ok=True)
        return self.runtime.process(pid)

    def listener(self,pid=70001,inode=991,owned=True):
        # Kernel /proc/net/tcp field order: sl local remote state queues timer retr uid timeout inode.
        (self.proc/'net/tcp').write_text(f'header\n 0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000 {self.uid} 0 {inode}\n')
        if owned: (self.proc/str(pid)/'fd/10').symlink_to(f'socket:[{inode}]')

    def serving(self,canvas=256):
        p=self.process(args=self.args(canvas));self.listener()
        self.runtime.pidfile.write_text(str(p['pid'])+'\n');return p


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name);self.r=self.f.runtime

    def test_exact_attestation_and_fast_matching_idle(self):
        self.f.serving(128)
        status=self.r.status()
        self.assertTrue(status['ready']);self.assertTrue(status['verified'])
        self.assertEqual(status['canvas_length'],128);self.assertEqual(status['pid'],70001)
        self.assertEqual(status['starttime'],'100');self.assertEqual(len(status['config_hash']),64)
        with mock.patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            ensured=self.r.ensure(128)
        self.assertFalse(ensured['changed']);self.assertEqual(ensured['config_hash'],status['config_hash'])

    def test_smallest_canvas_transition_and_new_lengths(self):
        self.f.serving(256)
        self.f.process(70003,['python','training.py'],parent=1,start='103',state='T')
        signals=[]
        def terminate(api,engine=False):
            signals.append((api['pid'],engine))
            shutil.rmtree(self.f.proc/str(api['pid']))
            (self.f.proc/'net/tcp').write_text('header\n')
        def launch(args,**kwargs):
            self.assertEqual(args,['/bin/bash',str(self.r.launcher),'diffusion'])
            self.assertEqual(kwargs['env']['CANVAS_LENGTH'],'8')
            self.assertEqual(kwargs['env']['ENABLE_TORCH_PROFILER'],'1')
            self.assertTrue(kwargs['start_new_session'])
            self.f.process(70002,args=self.f.args(8),start='200')
            self.f.listener(pid=70002)
            return types.SimpleNamespace(pid=70002)
        with mock.patch.object(self.r,'validate_launcher'), \
             mock.patch.object(self.r,'terminate',side_effect=terminate), \
             mock.patch.object(m.subprocess,'Popen',side_effect=launch), \
             mock.patch.object(m.time,'sleep'):
            result=self.r.transition(8,'smallest-canvas')
        self.assertEqual(signals,[(70001,False)])
        self.assertTrue(result['ready']);self.assertTrue(result['verified'])
        self.assertTrue(result['changed']);self.assertEqual(result['canvas_length'],8)
        self.assertEqual(result['pid'],70002);self.assertEqual(result['starttime'],'200')
        self.assertEqual(self.r.process(70003)['state'],'T')
        for canvas in (16,32):
            with self.subTest(canvas=canvas):
                current=self.f.process(70002,args=self.f.args(canvas),start='200')
                attested=self.r.wait_ready(current,canvas,time.monotonic()+1,changed=True)
                self.assertEqual(attested['canvas_length'],canvas)
                self.assertTrue(attested['ready']);self.assertTrue(attested['verified'])

    def test_invalid_canvas_configuration_is_not_attested(self):
        for canvas in (0,7,12,1024,True,'8'):
            with self.subTest(canvas=canvas):
                process=self.f.process(args=self.f.args(canvas))
                with self.assertRaisesRegex(m.Refusal,'Cannot attest owned runtime configuration'):
                    self.r.api(process)

    def test_identity_refusal_matrix(self):
        original=self.f.serving()
        for flag,bad in [('--revision','other'),('--served-model-name','other'),('--host','0.0.0.0'),('--port','8001')]:
            args=self.f.args();args[args.index(flag)+1]=bad
            self.f.process(args=args)
            with self.assertRaises(m.Refusal):self.r.status()
        args=self.f.args();args[3]='different/checkpoint';self.f.process(args=args)
        with self.assertRaises(m.Refusal):self.r.status()
        args=self.f.args();args[1]='/unmanaged/bin/vllm';self.f.process(args=args)
        with self.assertRaises(m.Refusal):self.r.status()
        self.f.process(uid=self.f.uid+1)
        with self.assertRaises(m.Refusal):self.r.status()
        self.f.process(args=['python','training.py'],start='222',state='T')
        with self.assertRaises(m.Refusal):self.r.status()
        self.assertIsNone(self.r.live_identity(original))

    def test_port_refusal_even_with_valid_pid(self):
        self.f.process();self.f.listener(owned=False)
        with self.assertRaisesRegex(m.Refusal,'unmanaged listener'):self.r.status()
        shutil.rmtree(self.f.proc/'70001')
        with self.assertRaisesRegex(m.Refusal,'without a verified'):self.r.status()

    def test_busy_and_unknown_metrics_refuse_matching_ensure(self):
        self.f.serving()
        for metrics in [self.f.metrics.replace(' 0.0',' 1.0',1),'',
                        self.f.metrics.replace(' 0.0',' NaN',1),
                        self.f.metrics.replace(m.SERVED,'wrong-model')]:
            self.f.metrics=metrics
            with self.assertRaises(m.Refusal):self.r.ensure(256)
        self.assertFalse(self.r.statefile.exists())

    def test_busy_transition_never_signals_or_launches(self):
        self.f.serving()
        self.f.metrics=self.f.metrics.replace(' 0.0',' 1.0',1)
        with mock.patch.object(self.r,'validate_launcher'), \
             mock.patch.object(self.r,'terminate',side_effect=AssertionError('must not signal')), \
             mock.patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(m.Refusal,'Refusing restart'):
                self.r.transition(128,'operation')

    def test_launcher_validation_before_any_stop(self):
        self.f.serving()
        self.r.launcher.write_text('# old launcher without canvas support\n')
        with mock.patch.object(self.r,'terminate',side_effect=AssertionError('must never signal')):
            with self.assertRaisesRegex(m.Refusal,'CANVAS_LENGTH'):
                self.r.transition(128,'operation')
        self.assertEqual(self.r.find_api()['canvas_length'],256)

    def test_fl_lock_blocks_even_matching_fast_path(self):
        self.f.serving()
        fd=os.open(self.r.lockfile,os.O_RDWR|os.O_CREAT,0o600)
        self.addCleanup(os.close,fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m.atomic_json(self.r.statefile,{'operation_id':'op','target_canvas':128,'phase':'queued'})
        with self.assertRaisesRegex(m.Refusal,'different canvas'):
            self.r.ensure(256)
        with mock.patch.object(self.r,'wait_operation',return_value={'joined':True}) as waiter:
            self.assertEqual(self.r.ensure(128),{'joined':True})
            waiter.assert_called_once_with('op',128)

    def test_pidfd_verifies_current_identity_before_signal(self):
        api=self.r.api(self.f.serving())
        calls=[]
        with mock.patch.object(m.os,'pidfd_open',return_value=123,create=True), \
             mock.patch.object(m.signal,'pidfd_send_signal',side_effect=lambda *a:calls.append(a),create=True), \
             mock.patch.object(m.os,'close'):
            self.r.terminate(api)
            self.assertEqual(calls,[(123,m.signal.SIGTERM,None,0)])
            self.f.process(start='101',args=['python','training.py'],state='T')
            self.r.terminate(api)
            self.assertEqual(len(calls),1)
        self.assertNotIn('SIGKILL',(HERE.parent/'scripts/canvas-runtime.py').read_text().split('from __future__')[1])

    def test_descendant_engine_only_training_excluded(self):
        api=self.r.api(self.f.serving())
        self.f.process(70002,['VLLM::EngineCore'],parent=70001,start='102')
        self.f.process(70003,['python','training.py'],parent=1,start='103',state='T')
        self.f.process(70004,['VLLM::EngineCore'],parent=1,start='104')
        self.f.process(70005,['python','resource_tracker'],parent=70001,start='105')
        self.assertEqual([x['pid'] for x in self.r.descendants(api)],[70002])

    def test_readiness_rechecks_shape_and_identity(self):
        api=self.f.serving(512)
        result=self.r.wait_ready(api,512,time.monotonic()+1,changed=True)
        self.assertTrue(result['changed']);self.assertEqual(result['canvas_length'],512)
        with self.assertRaisesRegex(m.Refusal,'differs from requested'):
            self.r.wait_ready(api,128,time.monotonic()+1,changed=True)
        self.f.process(start='200')
        with self.assertRaisesRegex(m.Refusal,'exited'):
            self.r.wait_ready(api,512,time.monotonic()+1,changed=True)

    def test_no_force_kill_when_graceful_exit_fails(self):
        self.f.serving()
        self.f.process(70002,['VLLM::EngineCore'],parent=70001,start='102')
        calls=[]
        with mock.patch.object(self.r,'validate_launcher'),mock.patch.object(self.r,'wait_gone',return_value=False), \
             mock.patch.object(self.r,'terminate',side_effect=lambda p,engine=False:calls.append((p['pid'],engine))), \
             mock.patch.object(m.time,'sleep'),mock.patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(m.Refusal,'no force kill'):
                self.r.transition(128,'op')
        self.assertEqual(calls,[(70001,False),(70002,True)])

    def test_recorded_launch_is_status_not_unmanaged_api(self):
        p=self.f.process(args=['/bin/bash',str(self.r.launcher),'diffusion'])
        self.r.pidfile.write_text(str(p['pid']))
        m.atomic_json(self.r.statefile,{'operation_id':'op','phase':'starting','target_canvas':128,'spawn':p})
        result=self.r.status();self.assertFalse(result['ready']);self.assertEqual(result['transition'],'starting')
        self.f.process(args=['/bin/bash','unrelated.sh'],start='101')
        with self.assertRaises(m.Refusal):self.r.status()

    def test_real_detached_child_retains_inherited_flock_after_parent_fd_close(self):
        lock=Path(self.temp.name)/'inherited.lock';ready=Path(self.temp.name)/'ready';release=Path(self.temp.name)/'release'
        fd=os.open(lock,os.O_RDWR|os.O_CREAT,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code='''import os,sys,time\nfrom pathlib import Path\nfd=int(sys.argv[1]);Path(sys.argv[2]).write_text(str(os.getsid(0)))\nend=time.monotonic()+5\nwhile not Path(sys.argv[3]).exists() and time.monotonic()<end:time.sleep(.01)\nos.close(fd)\n'''
        child=subprocess.Popen([sys.executable,'-c',code,str(fd),str(ready),str(release)],pass_fds=(fd,),start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and child.terminate())
        os.close(fd)  # Equivalent to the SSH frontend exiting/cancelling.
        end=time.monotonic()+2
        while not ready.exists() and time.monotonic()<end:time.sleep(.01)
        self.assertTrue(ready.exists());self.assertEqual(int(ready.read_text()),child.pid)
        second=os.open(lock,os.O_RDWR)
        try:
            with self.assertRaises(BlockingIOError):fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
            release.touch();self.assertEqual(child.wait(timeout=2),0)
            fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(second)


if __name__=='__main__':unittest.main(verbosity=2)
