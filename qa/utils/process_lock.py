# -*- coding: utf-8 -*-
import os
import sys
import time
import logging

logger = logging.getLogger("MedicalQA.ProcessLock")

def acquire_process_lock(lock_name: str):
    """
    利用锁文件和 PID 在 Windows/Unix 环境下实现单实例进程互斥。
    如果发现已有相同锁名的活跃进程在后台运行，会强行将其杀死关闭，确保独占执行。
    每个脚本退出时（不论正常或异常），都会通过 atexit 自动清除该锁文件。
    """
    # 锁文件存放在项目根目录下
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_file = os.path.join(root_dir, f"{lock_name}.lock")
    current_pid = os.getpid()

    # 1. 启动前检查：若存在旧锁文件，读取 PID 并尝试关闭
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r", encoding="utf-8") as f:
                old_pid_str = f.read().strip()
            if old_pid_str.isdigit():
                old_pid = int(old_pid_str)
                if old_pid != current_pid and is_pid_alive(old_pid):
                    logger.warning(f"⚠️ [进程锁冲突] 检测到相同实例已在后台运行 (PID: {old_pid})。正在强制关闭旧实例...")
                    kill_pid(old_pid)
                    # 适当等待旧进程彻底退场释放句柄与文件锁
                    time.sleep(1.5)
        except Exception as e:
            logger.error(f"⚠️ [进程锁检查] 读取或清理旧实例失败: {e}")

    # 2. 写入当前进程的 PID 以建立新锁
    try:
        with open(lock_file, "w", encoding="utf-8") as f:
            f.write(str(current_pid))
        logger.info(f"🔑 [进程锁锁定] 成功锁定单实例运行 (PID: {current_pid})，锁文件: {lock_file}")
    except Exception as e:
        logger.error(f"⚠️ [进程锁锁定] 写入锁文件失败: {e}")

    # 3. 注册安全退出清理钩子，退出时安全关闭/删除实例锁文件
    import atexit
    def release_lock():
        try:
            if os.path.exists(lock_file):
                with open(lock_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content == str(current_pid):
                    os.remove(lock_file)
                    logger.info(f"🔓 [进程锁释放] 当前实例 (PID: {current_pid}) 已安全退出并清除锁。")
        except Exception as e:
            pass

    atexit.register(release_lock)

def is_pid_alive(pid: int) -> bool:
    """利用标准库的 os.kill 信号检查在 Windows/Unix 下判断进程是否还处于活跃状态"""
    if pid <= 0:
        return False
    try:
        # 信号 0 在 Windows/Python 3.2+ 下可用于活跃性探测而不会真正发送破坏性信号
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def kill_pid(pid: int):
    """在 Windows/Unix 下强制杀死指定 PID 的进程"""
    try:
        # 在 Windows 上，信号 9 (SIGKILL) 是直接强制终止进程
        os.kill(pid, 9)
        logger.info(f"✨ [进程关闭成功] 已成功强制终止 PID {pid} 的旧实例进程。")
    except Exception as e:
        logger.error(f"⚠️ [进程关闭失败] 无法终止 PID {pid} 的进程，可能是权限不足或进程已消亡: {e}")
