from types import SimpleNamespace
import subprocess

from gradlab.hardware_metrics import HardwareSampler
from gradlab.metric_names import validate_metric_payload


def test_hardware_missing_sensors_are_omitted_and_cpu_is_primed(monkeypatch):
    monkeypatch.setattr('gradlab.hardware_metrics.shutil.which', lambda _: None)
    monkeypatch.setattr('gradlab.hardware_metrics.psutil.cpu_percent', lambda **_: 25)
    monkeypatch.setattr('gradlab.hardware_metrics.psutil.virtual_memory',
                        lambda: SimpleNamespace(total=1024**3, available=512*1024**2))
    sampler = HardwareSampler()
    first = sampler.sample()
    assert 'system/cpu_utilization_percentage' not in first
    assert first['system/system_memory_usage_megabytes'] == 512
    assert first['system/system_memory_usage_percentage'] == 50
    second = sampler.sample()
    assert second['system/cpu_utilization_percentage'] == 25
    assert not any(k.startswith('system/gpu/') for k in second)
    validate_metric_payload(second)


def test_gpu_timeout_and_unsupported_fields_do_not_fabricate_measurements(monkeypatch):
    monkeypatch.setattr('gradlab.hardware_metrics.shutil.which', lambda _: '/bin/nvidia-smi')
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == 1
        raise subprocess.TimeoutExpired('nvidia-smi', 1)
    monkeypatch.setattr('gradlab.hardware_metrics.subprocess.run', timeout)
    sampler = HardwareSampler()
    assert not any(k.startswith('system/gpu/') for k in sampler.sample())
    monkeypatch.setattr('gradlab.hardware_metrics.subprocess.run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout='0, 80, 4000, 24000, [N/A], 63\n1, nan, -1, 24000, 200, 65'))
    metrics = sampler.sample()
    assert metrics['system/gpu/0/utilization_percentage'] == 80
    assert 'system/gpu/0/power_usage_watts' not in metrics
    assert 'system/gpu/1/utilization_percentage' not in metrics
    assert 'system/gpu/1/memory_usage_megabytes' not in metrics
    validate_metric_payload(metrics)
