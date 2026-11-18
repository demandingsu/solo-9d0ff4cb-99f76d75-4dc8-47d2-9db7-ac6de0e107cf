"""波形交接台自测:重叠、补传、幂等、原子性、窗状态机、交错一致性。"""
import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app.core import SAMPLE_MS, WINDOW_SAMPLES
from app.main import create_app

W0 = 1_700_000_000_000  # 整 10 秒对齐的窗起点(epoch 毫秒)
STA = "STA1"


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def row(station, start_ms, rnd, no, samples):
    return f"{station},{start_ms},{rnd},{no},{' '.join(map(str, samples))}"


def upload(client, text):
    return client.post("/api/upload", content=text)


def publish(client, station=STA, wstart=W0):
    return client.post("/api/publish",
                       json={"station": station, "window_start_ms": wstart})


def picking(client, station=STA, wstart=W0):
    return client.get(f"/api/picking/{station}/{wstart}")


def windows(client, station=STA, lo=W0, hi=W0 + 10_000):
    r = client.get(f"/api/windows?station={station}&from_ms={lo}&to_ms={hi}")
    assert r.status_code == 200, r.text
    return r.json()["windows"]


def fill_window(client, rnd, value, wstart=W0, station=STA):
    """用一个整窗片段(1000 样点)铺满窗口并返回上传响应。"""
    return upload(client, row(station, wstart, rnd, 0, [value] * 1000))


# ---------------------------------------------------------------- 基础流程

def test_upload_publish_pick_flow(client):
    r = fill_window(client, rnd=1, value=7)
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1

    w = windows(client)[0]
    assert (w["filled"], w["gaps"], w["state"]) == (1000, 0, "none")

    p = publish(client)
    assert p.status_code == 200, p.text
    assert p.json()["version"] == 1

    pk = picking(client)
    assert pk.status_code == 200
    d = pk.json()
    assert d["samples"] == [7] * 1000
    assert d["rounds"] == [1] * 1000
    assert d["version"] == 1


def test_iso_timestamp_accepted_and_aligned(client):
    r = upload(client, f"{STA},2023-11-14T22:13:20.000Z,1,0,1 2 3")
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1


def test_start_must_align_10ms(client):
    r = upload(client, row(STA, W0 + 5, 1, 0, [1, 2, 3]))
    assert r.status_code == 400
    assert "10毫秒" in r.json()["detail"]
    r2 = upload(client, f"{STA},2023-11-14T22:13:20.005Z,1,0,1 2 3")
    assert r2.status_code == 400


# ---------------------------------------------------------------- 幂等与拒收

def test_idempotent_reupload_same_content(client):
    text = row(STA, W0, 1, 0, [1] * 1000)
    assert upload(client, text).json()["inserted"] == 1
    r = upload(client, text)
    assert r.status_code == 200
    assert r.json() == {"inserted": 0, "skipped": 1, "stale_windows": []}
    # 现役窗不因同内容重传而过期
    fill_window(client, 1, 1)  # 该键已存在,跳过
    publish(client)
    r = upload(client, text)
    assert r.json()["stale_windows"] == []
    assert picking(client).status_code == 200


def test_same_key_different_content_rejected_and_state_kept(client):
    fill_window(client, 1, 1)
    publish(client)
    before = picking(client).json()

    r = upload(client, row(STA, W0, 1, 0, [9] * 1000))  # 同轮同号异内容
    assert r.status_code == 400
    assert "同轮同号" in r.json()["detail"]
    # 上传失败不改变原有片段与窗状态
    assert picking(client).status_code == 200
    assert picking(client).json() == before


def test_same_round_overlap_conflict_within_batch(client):
    text = "\n".join([
        row(STA, W0, 1, 0, [1] * 100),
        row(STA, W0 + 500, 1, 1, [2] * 100),  # 与前一片重叠 50 样点且异值
    ])
    r = upload(client, text)
    assert r.status_code == 400
    assert "重叠异值" in r.json()["detail"]
    assert windows(client)[0]["filled"] == 0  # 整批拒收,一片未入


def test_same_round_overlap_conflict_against_db(client):
    assert upload(client, row(STA, W0, 1, 0, [1] * 100)).status_code == 200
    r = upload(client, row(STA, W0 + 500, 1, 1, [2] * 100))
    assert r.status_code == 400
    assert "重叠异值" in r.json()["detail"]
    assert windows(client)[0]["filled"] == 100  # 只有第一批


def test_same_round_overlap_same_value_allowed(client):
    assert upload(client, row(STA, W0, 1, 0, [5] * 100)).status_code == 200
    r = upload(client, row(STA, W0 + 500, 1, 1, [5] * 100))  # 重叠 50 样点同值
    assert r.status_code == 200
    assert windows(client)[0]["filled"] == 150


def test_upload_atomicity_on_parse_error(client):
    text = "\n".join([
        row(STA, W0, 1, 0, [1] * 100),
        row(STA, W0 + 1003, 1, 1, [2] * 100),  # 起点未按 10ms 对齐
    ])
    r = upload(client, text)
    assert r.status_code == 400
    assert windows(client)[0]["filled"] == 0  # 第一行也未入库


def test_failed_upload_keeps_segments_and_window(client):
    fill_window(client, 1, 1)
    publish(client)
    # 批内一行合法新片段 + 一行同键异内容 -> 整批拒收
    text = "\n".join([
        row(STA, W0 + 20_000, 2, 0, [3] * 100),
        row(STA, W0, 1, 0, [9] * 1000),
    ])
    assert upload(client, text).status_code == 400
    assert windows(client)[0]["state"] == "active"  # 窗未过期
    assert picking(client).json()["samples"] == [1] * 1000
    # 合法那行也未入库
    w2 = windows(client, lo=W0 + 20_000, hi=W0 + 30_000)[0]
    assert w2["filled"] == 0


# ---------------------------------------------------------------- 轮次覆盖规则

def test_higher_round_overwrites_overlap_and_marks_stale(client):
    fill_window(client, 1, 1)
    publish(client)
    assert picking(client).status_code == 200

    # 高轮次补传前 100 样点 -> 现役窗立即过期
    r = upload(client, row(STA, W0, 2, 0, [2] * 100))
    assert r.status_code == 200
    assert r.json()["stale_windows"] == [{"station": STA, "window_start_ms": W0}]
    assert windows(client)[0]["state"] == "stale"

    # 拾取接口不再把旧窗当现役
    pk = picking(client)
    assert pk.status_code == 409

    # 重新发布成功才替换
    p = publish(client)
    assert p.status_code == 200
    assert p.json()["version"] == 2
    d = picking(client).json()
    assert d["samples"] == [2] * 100 + [1] * 900
    assert d["rounds"] == [2] * 100 + [1] * 900


def test_late_lower_round_does_not_overwrite(client):
    fill_window(client, 5, 5)   # 高轮次先铺满
    publish(client)
    r = upload(client, row(STA, W0, 2, 0, [2] * 1000))  # 晚到的低轮次
    assert r.status_code == 200
    assert r.json()["stale_windows"] == []  # 有效样点未变,窗不过期
    d = picking(client).json()
    assert d["samples"] == [5] * 1000
    assert d["rounds"] == [5] * 1000


def test_lower_round_fills_only_gaps(client):
    # 轮次 5 覆盖样点 0..499
    assert upload(client, row(STA, W0, 5, 0, [5] * 500)).status_code == 200
    # 晚到的轮次 2 覆盖 250..749:250..499 不得覆盖高轮次,500..749 补缺
    assert upload(client, row(STA, W0 + 2500, 2, 0, [2] * 500)).status_code == 200
    # 轮次 1 覆盖 750..999
    assert upload(client, row(STA, W0 + 7500, 1, 0, [1] * 250)).status_code == 200

    w = windows(client)[0]
    assert (w["filled"], w["gaps"]) == (1000, 0)
    assert publish(client).status_code == 200
    d = picking(client).json()
    assert d["samples"] == [5] * 500 + [2] * 250 + [1] * 250
    assert d["rounds"] == [5] * 500 + [2] * 250 + [1] * 250


def test_higher_round_only_covers_its_own_positions(client):
    # 轮次 1 先铺满,轮次 3 只补传中间 100 点 -> 只覆盖那 100 点
    fill_window(client, 1, 1)
    upload(client, row(STA, W0 + 4000, 3, 0, [3] * 100))
    publish(client)
    d = picking(client).json()
    assert d["samples"] == [1] * 400 + [3] * 100 + [1] * 500
    assert d["rounds"] == [1] * 400 + [3] * 100 + [1] * 500


# ---------------------------------------------------------------- 发布与窗状态

def test_publish_incomplete_fails_and_creates_nothing(client):
    assert upload(client, row(STA, W0, 1, 0, [1] * 500)).status_code == 200
    p = publish(client)
    assert p.status_code == 409
    assert p.json()["filled"] == 500
    assert windows(client)[0]["state"] == "none"
    assert picking(client).status_code == 404


def test_picking_not_found_for_unknown_window(client):
    assert picking(client).status_code == 404


def test_stale_window_republish_restores_active(client):
    fill_window(client, 1, 1)
    publish(client)
    upload(client, row(STA, W0, 2, 0, [2] * 10))  # 改动 10 点 -> 过期
    assert windows(client)[0]["state"] == "stale"
    assert picking(client).status_code == 409
    assert publish(client).status_code == 200
    assert picking(client).status_code == 200
    assert picking(client).json()["version"] == 2


def test_backfill_outside_active_window_does_not_stale(client):
    fill_window(client, 1, 1)
    publish(client)
    # 补传到相邻窗,不影响现役窗
    r = upload(client, row(STA, W0 + 10_000, 1, 1, [9] * 100))
    assert r.json()["stale_windows"] == []
    assert windows(client)[0]["state"] == "active"


def test_window_detail_shows_gap_ranges(client):
    upload(client, row(STA, W0 + 2000, 1, 0, [1] * 100))  # 样点 200..299
    d = client.get(f"/api/windows/{STA}/{W0}").json()
    assert d["filled"] == 100
    assert d["gap_ranges"] == [[0, 199], [300, 999]]
    assert d["state"] == "none"


def test_multi_segment_assembly_across_windows(client):
    # 两段跨窗片段拼满两个窗
    text = "\n".join([
        row(STA, W0 + 5000, 1, 0, [4] * 1000),  # 覆盖 W0 后半与 W1 前半
        row(STA, W0, 1, 1, [3] * 500),          # W0 前半
        row(STA, W0 + 15_000, 1, 2, [5] * 500), # W1 后半
    ])
    assert upload(client, text).status_code == 200
    ws = windows(client, lo=W0, hi=W0 + 20_000)
    assert [w["filled"] for w in ws] == [1000, 1000]
    for w in (W0, W0 + 10_000):
        assert publish(client, wstart=w).status_code == 200
    d0 = picking(client, wstart=W0).json()
    d1 = picking(client, wstart=W0 + 10_000).json()
    assert d0["samples"] == [3] * 500 + [4] * 500
    assert d1["samples"] == [4] * 500 + [5] * 500


# ---------------------------------------------------------------- 交错一致性

def test_interleaved_upload_publish_read_consistency(client):
    """上传/发布/读取交错:读到的样点、轮次、窗状态须来自同一已提交状态。

    不变式:值 == 轮次(轮次1填1,轮次2填2)。若 samples 与 rounds 来自
    不同提交状态,会出现 samples[i] != rounds[i]。
    """
    fill_window(client, 1, 1)
    publish(client)
    errors = []
    stop = threading.Event()

    def writer():
        try:
            for _ in range(25):
                r = upload(client, row(STA, W0, 2, 0, [2] * 1000))
                assert r.status_code == 200
                p = publish(client)
                assert p.status_code == 200
        except Exception as e:  # noqa: BLE001
            errors.append(f"writer: {e}")
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                resp = picking(client)
                if resp.status_code == 200:
                    d = resp.json()
                    assert len(d["samples"]) == 1000
                    assert len(d["rounds"]) == 1000
                    assert all(s == r for s, r in zip(d["samples"], d["rounds"]))
                    assert d["state"] == "active"
                elif resp.status_code == 409:
                    pass  # 过期是合法中间态
                else:
                    errors.append(f"reader: unexpected {resp.status_code}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"reader: {e}")

    threads = [threading.Thread(target=writer)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    # 收尾:轮次 2 已生效
    d = picking(client).json()
    assert d["samples"] == [2] * 1000
    assert d["rounds"] == [2] * 1000


# ---------------------------------------------------------------- 跨实例(跨连接)一致性
#
# 两个服务实例 = 同一 SQLite 文件上的两个独立连接。下列测试用第二个
# 原始连接扮演"另一台实例",在查询请求的多次 SELECT 之间插入已提交的
# 上传/发布,验证缺口列表与详情返回的统计、样点/轮次、窗状态与版本
# 必须来自同一个已提交状态。

@pytest.fixture()
def peer_conn(tmp_path):
    """扮演另一台服务实例的独立连接(独立进程即独立连接)。

    check_same_thread=False:确定性测试的注入钩子运行在 TestClient 的
    portal 线程,而连接在测试主线程创建;真实多进程场景各自单线程连接,
    不受此参数影响。
    """
    other = sqlite3.connect(str(tmp_path / "test.db"),
                            isolation_level=None, check_same_thread=False)
    other.row_factory = sqlite3.Row
    other.execute("PRAGMA busy_timeout=5000")
    yield other
    other.close()


def _peer_insert(other, rnd, no, start_ms, values):
    other.execute("BEGIN IMMEDIATE")
    other.execute(
        "INSERT INTO segments(station,round,segment_no,start_ms,n,samples,"
        "content_hash,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (STA, rnd, no, start_ms, len(values), json.dumps(values),
         f"h{rnd}-{no}", "t"))
    other.commit()


def _peer_publish(other, version, values, rounds):
    other.execute("BEGIN IMMEDIATE")
    other.execute(
        "INSERT INTO windows(station,window_start_ms,state,samples,rounds,"
        "version,published_at) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(station,window_start_ms) DO UPDATE SET "
        "state='active', samples=excluded.samples, rounds=excluded.rounds, "
        "version=excluded.version, published_at=excluded.published_at",
        (STA, W0, "active", json.dumps(values), json.dumps(rounds),
         version, "t"))
    other.commit()


def test_list_query_pins_single_snapshot_across_peer_commit(client, peer_conn):
    """列表查询两次 SELECT 之间,另一实例补齐并发布:不得报"现役窗只有500点"。"""
    # 本实例先入库 500 点(窗未发布)
    assert upload(client, row(STA, W0, 1, 0, [1] * 500)).status_code == 200

    # 让对端的提交恰好发生在列表请求的 segments 查询之后、windows 查询之前
    orig_fetch = __import__("app.core", fromlist=["_fetch_segments"])._fetch_segments

    def fetch_then_peer_commits(conn, station, lo_ms, hi_ms):
        out = orig_fetch(conn, station, lo_ms, hi_ms)
        _peer_insert(peer_conn, 1, 1, W0 + 500 * SAMPLE_MS, [1] * 500)
        _peer_publish(peer_conn, 1, [1] * 1000, [1] * 1000)
        return out

    __import__("app.core", fromlist=["_fetch_segments"])._fetch_segments = fetch_then_peer_commits
    try:
        w = windows(client)[0]
    finally:
        __import__("app.core", fromlist=["_fetch_segments"])._fetch_segments = orig_fetch

    # 同一响应内的统计与状态必须自洽:现役窗必然 1000 点;
    # 未发布(none)则必须仍是发布前的 500 点。两种都是合法的单一快照
    if w["state"] == "active":
        assert w["filled"] == 1000 and w["gaps"] == 0 and w["version"] == 1
    else:
        assert w["state"] == "none"
        assert w["filled"] == 500 and w["gaps"] == 500 and w["version"] == 0
    # 对端提交结束后,新请求必须看到最新已提交状态
    latest = windows(client)[0]
    assert latest["filled"] == 1000 and latest["gaps"] == 0
    assert latest["state"] == "active" and latest["version"] == 1


def test_detail_query_pins_single_snapshot_across_peer_commit(client, peer_conn):
    """详情查询的样点/轮次与窗状态必须同快照;补传使现役窗过期时不得混报。"""
    fill_window(client, 1, 1)
    publish(client)  # 现役 v1:1000 点全来自轮次 1

    orig_fetch = __import__("app.core", fromlist=["_fetch_segments"])._fetch_segments

    def fetch_then_peer_stales(conn, station, lo_ms, hi_ms):
        out = orig_fetch(conn, station, lo_ms, hi_ms)
        # 另一实例高轮次补传前 100 点 -> 现役窗立即过期
        _peer_insert(peer_conn, 2, 0, W0, [2] * 100)
        peer_conn.execute("BEGIN IMMEDIATE")
        peer_conn.execute(
            "UPDATE windows SET state='stale' WHERE station=? AND window_start_ms=?",
            (STA, W0))
        peer_conn.commit()
        return out

    core_mod = __import__("app.core", fromlist=["_fetch_segments"])
    core_mod._fetch_segments = fetch_then_peer_stales
    try:
        d = client.get(f"/api/windows/{STA}/{W0}").json()
    finally:
        core_mod._fetch_segments = orig_fetch

    # 旧快照:active + 旧样点(全 1,全轮次 1);新快照:stale + 新样点(前100为2)
    assert d["filled"] + d["gaps"] == WINDOW_SAMPLES
    assert d["state"] in ("active", "stale")
    if d["state"] == "active":
        assert d["version"] == 1
        assert d["samples"] == [1] * 1000
        assert d["rounds"] == [1] * 1000
    else:
        assert d["samples"][:100] == [2] * 100
        assert d["rounds"][:100] == [2] * 100
        assert d["samples"][100:] == [1] * 900
        assert d["rounds"][100:] == [1] * 900
    # 再查一次必然是新快照
    d2 = client.get(f"/api/windows/{STA}/{W0}").json()
    assert d2["state"] == "stale"
    assert d2["samples"][:100] == [2] * 100


def test_read_transaction_does_not_block_peer_writer(client, peer_conn):
    """查询持有读快照期间,另一实例的上传仍须及时完成(WAL 读写不互斥)。"""
    fill_window(client, 1, 1)
    publish(client)
    write_finished = threading.Event()

    def peer_write():
        _peer_insert(peer_conn, 2, 0, W0, [2] * 100)  # 独立连接上的写事务
        write_finished.set()

    core_mod = __import__("app.core", fromlist=["_fetch_segments"])
    orig_fetch = core_mod._fetch_segments

    def hold_snapshot(conn, station, lo_ms, hi_ms):
        out = orig_fetch(conn, station, lo_ms, hi_ms)
        t = threading.Thread(target=peer_write)
        t.start()
        # 读事务尚未提交:给对端写事务留出充分时间完成
        t.join(timeout=2.0)
        return out

    core_mod._fetch_segments = hold_snapshot
    try:
        d = client.get(f"/api/windows/{STA}/{W0}").json()
    finally:
        core_mod._fetch_segments = orig_fetch
    assert d["state"] == "active"  # 读请求锚定写之前的快照,自洽即可
    # 对端写事务在本读事务持有期间就已完成,未被读锁阻塞
    assert write_finished.is_set()
    # 读事务结束后,新请求可见对端已提交的 100 个高轮次样点(窗状态由
    # 应用层上传流程维护,裸连接插入不会触发,这里只验证提交可见性)
    latest = client.get(f"/api/windows/{STA}/{W0}").json()
    assert latest["rounds"][:100] == [2] * 100
    assert latest["samples"][:100] == [2] * 100
    assert latest["state"] == "active"


def test_two_app_instances_interleaved_consistency(tmp_path):
    """两个独立 FastAPI 应用(两个连接,等价于两台实例)交错:
    一台持续补传+发布,另一台及其自身并发查询列表/详情,全程不得出现
    混合状态;写操作在查询压力下仍全部成功。"""
    db_path = str(tmp_path / "shared.db")
    app_a = create_app(db_path)
    app_b = create_app(db_path)
    with TestClient(app_a) as ca, TestClient(app_b) as cb:
        # 初始:轮次 1 铺满并由实例 B 发布
        assert upload(ca, row(STA, W0, 1, 0, [1] * 1000)).status_code == 200
        assert publish(cb).status_code == 200

        errors = []
        stop = threading.Event()

        def check_response(d):
            # 不变式 1:统计自洽
            if d["filled"] + d["gaps"] != WINDOW_SAMPLES:
                errors.append(f"filled+gaps={d['filled'] + d['gaps']}")
            # 不变式 2:样点与逐样点轮次一一对应(值==轮次),长度齐全
            if len(d["samples"]) != WINDOW_SAMPLES or len(d["rounds"]) != WINDOW_SAMPLES:
                errors.append("samples/rounds 长度异常")
            if any(s != r for s, r in zip(d["samples"], d["rounds"])
                   if r is not None):
                errors.append("samples 与 rounds 来自不同提交状态")
            # 不变式 3:已发布窗(active/stale)的有效样点必须是 1000,
            # 绝不允许"现役窗只有 500 点"这类混合报告
            if d["state"] in ("active", "stale") and d["filled"] != 1000:
                errors.append(f"{d['state']} 窗 filled={d['filled']}(混合状态)")
            if d["state"] in ("active", "stale") and d["version"] < 1:
                errors.append("已发布窗版本号异常")
            # 不变式 4:缺口区间与缺口数一致
            covered = sum(b - a + 1 for a, b in d["gap_ranges"])
            if covered != d["gaps"]:
                errors.append("gap_ranges 与 gaps 不一致")

        def writer():
            try:
                for rnd in range(2, 22):  # 20 轮补传+重新发布
                    r = upload(ca, row(STA, W0, rnd, 0, [rnd] * 1000))
                    assert r.status_code == 200, r.text
                    p = publish(cb, wstart=W0)  # 发布走另一实例
                    assert p.status_code == 200, p.text
                    assert p.json()["version"] == rnd
            except Exception as e:  # noqa: BLE001
                errors.append(f"writer: {e}")
            finally:
                stop.set()

        def reader(client, name):
            try:
                while not stop.is_set():
                    lw = windows(client)
                    for w in lw:
                        # 列表项也满足已发布窗必满点的自洽性
                        if w["state"] in ("active", "stale") and w["filled"] != 1000:
                            errors.append(f"{name} 列表混合状态: {w}")
                        if w["filled"] + w["gaps"] != 1000:
                            errors.append(f"{name} 列表统计不自洽: {w}")
                    d = client.get(f"/api/windows/{STA}/{W0}")
                    if d.status_code == 200:
                        check_response(d.json())
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader, args=(ca, f"A{i}")) for i in range(2)]
        threads += [threading.Thread(target=reader, args=(cb, f"B{i}")) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "线程超时未结束"
        assert not errors, errors[:5]

        # 收尾状态:实例 A 看到最终轮次 21 的现役窗
        d = cb.get(f"/api/picking/{STA}/{W0}").json()
        assert d["state"] == "active" and d["version"] == 21
        assert d["samples"] == [21] * 1000 and d["rounds"] == [21] * 1000
