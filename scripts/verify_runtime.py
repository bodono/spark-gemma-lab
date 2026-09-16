#!/usr/bin/env python3
"""Validate the native runtime, including a narrowly scoped NVIDIA wheel-tag defect."""
import ctypes,importlib.metadata as metadata,json,platform,struct,subprocess,sys
from pathlib import Path
check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
expected='nvidia-cusparselt-cu13 0.8.1 is not supported on this platform'
exception=None
if check.returncode:
    if check.stdout.strip()!=expected or check.stderr.strip() or platform.machine()!='aarch64':
        sys.stderr.write(check.stdout+check.stderr);raise SystemExit(check.returncode)
    dist=metadata.distribution('nvidia-cusparselt-cu13')
    if dist.version!='0.8.1' or 'Tag: py3-none-manylinux2014_sbsa' not in (dist.read_text('WHEEL') or ''):
        raise RuntimeError('Unexpected CUDA dependency metadata; refusing exception')
    libraries=[dist.locate_file(f) for f in dist.files if str(f).endswith('/libcusparseLt.so.0')]
    if len(libraries)!=1:raise RuntimeError('Cannot identify the cuSPARSELt library')
    with open(libraries[0],'rb') as f:header=f.read(20)
    if header[:6]!=b'\x7fELF\x02\x01' or struct.unpack('<H',header[18:20])[0]!=183:
        raise RuntimeError('cuSPARSELt is not an ELF64 little-endian AArch64 library')
    ctypes.CDLL(str(libraries[0]))
    exception={'package':'nvidia-cusparselt-cu13','version':'0.8.1','issue':'Vendor internal WHEEL tag uses sbsa instead of aarch64','checks':['exact pip-check output','exact version and wheel tag','ELF64 AArch64','dynamic library load'],'source':'https://pypi.org/project/nvidia-cusparselt-cu13/0.8.1/'}
import torch,vllm
report={'hostname':platform.node(),'architecture':platform.machine(),'python':platform.python_version(),'torch':torch.__version__,'vllm':vllm.__version__,'cuda_build':torch.version.cuda,'dependency_check':'pass with validated vendor metadata exception' if exception else 'pass','metadata_exception':exception,'gpu_inference_verified':False}
output=Path(sys.prefix).parent/'runtime-validation.json';output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
