"""Durable per-episode evaluation records shared by the real runners."""
import csv
import math
from datetime import datetime, timezone
import os
from pathlib import Path
import select
import sys
import tempfile
import time


class EpisodeResults:
    fields = ['episode', 'task', 'condition', 'executed', 'started_at',
              'duration_s', 'success', 'status']

    def __init__(self, path, *, task, condition, executed):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = dict(task=task, condition=condition, executed=int(executed))
        self.rows = []
        try:
            with self.path.open('x', newline='') as f:
                csv.writer(f).writerow(self.fields)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            with self.path.open(newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != self.fields:
                    raise ValueError(f'CSV columns do not match evaluation format: {self.path}')
                self.rows = list(reader)
            seen = set()
            for row in self.rows:
                if set(row) != set(self.fields) or any(v is None for v in row.values()):
                    raise ValueError(f'Incomplete CSV row: {self.path}')
                if any(row[key] != str(value) for key, value in self.meta.items()):
                    raise ValueError(f'CSV task/condition/executed does not match this run: {self.path}')
                number = int(row['episode'])
                if number < 1 or number in seen:
                    raise ValueError(f'Invalid or duplicate episode number: {number}')
                seen.add(number)
                if row['status'] == 'completed':
                    if row['success'] not in ('0', '1') or not math.isfinite(float(row['duration_s'])) or float(row['duration_s']) < 0:
                        raise ValueError(f'Invalid completed result: episode {number}')
        self.episode_offset = max((int(row['episode']) for row in self.rows), default=0)
        self.started = None
        print(f'[results] {self.path}；已有 {len(self.rows)} 条记录，'
              f'下一次编号 {self.episode_offset + 1}', flush=True)

    def save(self):
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + '.')
        try:
            with os.fdopen(fd, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def start(self, episode):
        self.rows.append(dict(self.meta, episode=self.episode_offset + episode,
                             started_at=datetime.now(timezone.utc).isoformat(),
                             duration_s='', success='', status='running'))
        self.save()
        print(f"[results] 开始记录编号 {self.rows[-1]['episode']}", flush=True)
        self.started = time.monotonic()

    def end(self, reason='enter'):
        if self.started is None or self.rows[-1]['status'] != 'running':
            return
        elapsed = time.monotonic() - self.started
        self.rows[-1].update(duration_s=f'{elapsed:.6f}',
                             status='pending_result' if reason == 'enter' else reason)
        self.save()

    def set_result(self, value):
        if value not in ('0', '1'):
            raise ValueError('result must be 0 or 1')
        if not self.rows or self.rows[-1]['status'] != 'pending_result':
            raise ValueError('no Enter-ended episode awaiting a result')
        self.rows[-1].update(success=value, status='completed')
        self.save()
        completed = [r for r in self.rows if r['status'] == 'completed']
        rate = sum(int(r['success']) for r in completed) / len(completed)
        successful = [r for r in completed if r['success'] == '1']
        mean_text = (f"{sum(float(r['duration_s']) for r in successful) / len(successful):.3f}s"
                     if successful else '暂无成功样本')
        print(f'[results] 已保存 {self.path}；已评估 {len(completed)} 次，'
              f'成功率 {rate:.1%}，成功尝试平均耗时 {mean_text}', flush=True)

    def prompt(self, stopped):
        if not self.rows or self.rows[-1]['status'] != 'pending_result':
            return
        print(f"本轮耗时 {self.rows[-1]['duration_s']}s。输入 1（成功）或 0（失败），然后 Enter：", flush=True)
        while not stopped():
            if not select.select([sys.stdin], [], [], .1)[0]:
                continue
            # Read bytes to avoid TextIOWrapper read-ahead interfering with select.
            data = bytearray()
            while True:
                char = os.read(sys.stdin.fileno(), 1)
                if not char:
                    raise EOFError('Deployment terminal closed')
                if char == b'\n':
                    break
                data.extend(char)
            value = data.decode().strip()
            if value in ('0', '1'):
                self.set_result(value)
                return
            print('请输入 1 或 0，然后 Enter：', flush=True)
