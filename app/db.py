"""SQLite 存储层:单连接 + 可重入锁,显式事务边界。

多实例共享同一个 SQLite 文件时的并发模型:
- 写事务一律 BEGIN IMMEDIATE,跨进程靠 SQLite 的库级写锁串行化,
  busy_timeout 让锁等待而不是立刻报 SQLITE_BUSY;
- 读请求包在单个延迟读事务(plain BEGIN,仅 SELECT)里,WAL 模式下
  读锁不阻塞写锁——查询期间其他实例仍可上传与发布;同时整个请求
  只锚定一个快照,多次 SELECT 看到的都是同一个已提交状态,不会把
  旧样点和新窗状态(或反之)拼成混合报告;
- 进程内的 RLock 只负责让本进程唯一连接上的事务不被其他线程嵌套打断,
  跨进程互斥由 SQLite 文件锁保证。
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    station      TEXT    NOT NULL,
    round        INTEGER NOT NULL,
    segment_no   INTEGER NOT NULL,
    start_ms     INTEGER NOT NULL,          -- UTC 起点,epoch 毫秒,须按 10ms 对齐
    n            INTEGER NOT NULL,          -- 样点数
    samples      TEXT    NOT NULL,          -- JSON 整数数组
    content_hash TEXT    NOT NULL,          -- 起点+振幅序列的散列,用于同内容判重
    created_at   TEXT    NOT NULL,
    PRIMARY KEY (station, round, segment_no)
);
CREATE INDEX IF NOT EXISTS idx_segments_station_range
    ON segments (station, start_ms);

CREATE TABLE IF NOT EXISTS windows (
    station         TEXT    NOT NULL,
    window_start_ms INTEGER NOT NULL,       -- UTC 整 10 秒窗起点,epoch 毫秒
    state           TEXT    NOT NULL,       -- 'active' | 'stale'
    samples         TEXT,                   -- 发布时的 1000 个整数样点(JSON)
    rounds          TEXT,                   -- 每个样点的来源轮次(JSON)
    version         INTEGER NOT NULL,
    published_at    TEXT,
    PRIMARY KEY (station, window_start_ms)
);
"""


class Database:
    def __init__(self, path: str):
        # isolation_level=None:关闭隐式事务,事务边界完全由本模块显式控制
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        # busy_timeout 必须先设置,切换 WAL 本身也可能遇到跨进程锁
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL 同步在 WAL 下兼顾安全与吞吐;跨进程读不阻塞写
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    @contextmanager
    def read_transaction(self):
        """延迟读事务:整个请求共享一个 WAL 快照。

        只包含 SELECT,不获取写锁,因此不会阻塞其他实例的上传/发布。
        """
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    @contextmanager
    def write_transaction(self):
        """立即写事务:跨进程互斥,整请求原子提交或整体回滚。"""
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def close(self):
        with self.lock:
            self.conn.close()
