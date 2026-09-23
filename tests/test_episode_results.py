import csv

import pytest

from threading_real import episode_results as module


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize('seconds', [30., 50.])
def test_episode_deadline_boundary_and_reset(monkeypatch, seconds):
    now = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    timer = module.EpisodeDeadline(seconds)
    assert not timer.expired()
    timer.start()
    now[0] += seconds - .1
    assert not timer.expired()
    now[0] = 100. + seconds
    assert timer.expired()
    assert timer.remaining() == 0
    timer.start()
    assert not timer.expired()
    assert timer.remaining() == seconds


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf')])
def test_episode_deadline_rejects_invalid_limits(seconds):
    with pytest.raises(ValueError, match='episode-timeout'):
        module.EpisodeDeadline(seconds)


def test_timeout_is_saved_as_failure_without_result_prompt(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    r = module.EpisodeResults(tmp_path / 'run.csv', task='maze', condition='selector', executed=True)
    r.start(1)
    now[0] = 150.
    r.end('timeout')
    r.end('max_cycles')
    r.end('interrupted')
    monkeypatch.setattr(module.select, 'select', lambda *a: pytest.fail('timeout must not prompt'))
    r.prompt(lambda: False)
    row = read(r.path)[0]
    assert row['status'] == 'timeout'
    assert row['success'] == '0'
    assert row['duration_s'] == '50.000000'


def test_duration_saved_before_prompt_and_result_survives_new_episode(tmp_path, monkeypatch):
    clock = iter([100., 112.5, 200., 202.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    path = tmp_path / 'run.csv'
    r = module.EpisodeResults(path, task='maze', condition='selector', executed=True)
    r.start(1)
    assert read(path)[0]['status'] == 'running'
    r.end()
    row = read(path)[0]
    assert row['duration_s'] == '12.500000'
    assert row['success'] == ''
    assert row['status'] == 'pending_result'
    r.end('interrupted')  # Stopping/cleanup must not replace the Enter timestamp.
    with pytest.raises(ValueError):
        r.set_result('yes')
    r.set_result('1')
    r.start(2)
    r.end('interrupted')
    rows = read(path)
    assert rows[0]['success'] == '1'
    assert rows[0]['duration_s'] == '12.500000'
    assert rows[1]['status'] == 'interrupted'
    assert rows[1]['success'] == ''
    resumed = module.EpisodeResults(path, task='maze', condition='selector', executed=True)
    assert resumed.episode_offset == 2
    assert read(path) == rows


def test_failed_atomic_replace_preserves_previous_csv(tmp_path, monkeypatch):
    path = tmp_path / 'run.csv'
    r = module.EpisodeResults(path, task='threading', condition='no_selector', executed=False)
    r.start(1)
    r.end()
    before = path.read_bytes()
    def fail(*args):
        raise OSError('disk error')
    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError, match='disk error'):
        r.set_result('0')
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_prompt_retries_invalid_input_and_saves_zero(tmp_path, monkeypatch):
    path = tmp_path / 'run.csv'
    r = module.EpisodeResults(path, task='maze', condition='selector', executed=True)
    r.start(1)
    r.end()
    read_fd, write_fd = module.os.pipe()
    module.os.write(write_fd, b'wrong\n0\n')
    module.os.close(write_fd)
    with module.os.fdopen(read_fd) as terminal:
        monkeypatch.setattr(module.sys, 'stdin', terminal)
        r.prompt(lambda: False)
    assert read(path)[0]['success'] == '0'
    assert read(path)[0]['status'] == 'completed'


def test_mean_duration_uses_only_successes(tmp_path, monkeypatch, capsys):
    clock = iter([0., 100., 200., 210., 300., 320.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    r = module.EpisodeResults(tmp_path / 'run.csv', task='maze',
                              condition='selector', executed=True)
    r.start(1)
    r.end()
    r.set_result('0')
    assert '暂无成功样本' in capsys.readouterr().out
    for episode in (2, 3):
        r.start(episode)
        r.end()
        r.set_result('1')
    output = capsys.readouterr().out
    assert '成功率 66.7%' in output
    assert '成功尝试平均耗时 15.000s' in output
    assert read(r.path)[0]['duration_s'] == '100.000000'


def test_resume_retains_history_numbers_and_statistics(tmp_path, monkeypatch, capsys):
    clock = iter([0., 10., 20., 40., 50.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    path = tmp_path / 'run.csv'
    first = module.EpisodeResults(path, task='maze', condition='selector', executed=True)
    first.start(1)
    first.end()
    first.set_result('1')
    original = read(path)[0]
    resumed = module.EpisodeResults(path, task='maze', condition='selector', executed=True)
    resumed.start(1)
    resumed.end()
    resumed.set_result('0')
    output = capsys.readouterr().out
    assert '成功率 50.0%' in output
    assert '成功尝试平均耗时 10.000s' in output
    resumed.start(2)
    assert [r['episode'] for r in read(path)] == ['1', '2', '3']
    assert read(path)[0] == original
    again = module.EpisodeResults(path, task='maze', condition='selector', executed=True)
    assert again.episode_offset == 3  # Preserve unfinished attempts as well.
    before = path.read_bytes()
    with pytest.raises(ValueError, match='does not match'):
        module.EpisodeResults(path, task='maze', condition='no_selector', executed=True)
    assert path.read_bytes() == before
