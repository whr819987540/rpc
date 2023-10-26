# 将go提供的HTTP接口封装为接口库
# 并进行状态管理

# 分布式学习框架
# 需要参照pytorch分布式中通信接口的定义
# 1、server需要知道有多少个client，client知道server的ip地址
# 2、server通过42069端口接收client的连接，并维持长连接。（TCP）
# 3、distribution阶段，server生成metainfo（server调用create_torrent）后，将metainfo推送给各个client
# 4、client收到metainfo后，开始下载（client调用start_downloading）。下载完成后，仍然进行seeding
# 5、client完成下载后，train
#   client告知server完成下载，当所有client都完成本轮下载后，4中seeding的内容不再有效，server告知client可以卸载某个metainfo
#   设计理念：在保证高带宽利用率的同时，降低内存利用率。
# 6、client开始回传
# 7、server收到所有client的回传后，完成aggregation。
# 回到3

import os
import re
import json
from types import SimpleNamespace
from subprocess import Popen
from threading import Thread
import requests
from time import sleep
import signal
import sys







def loadConfig(filepath: str):
    """
        加载jsonc配置文件
    """
    # 加载jsonc文件
    content = readJsonc(filepath)
    # 转换成结构体
    config = json.loads(content)
    # 转换成namespace
    config = to_namespace(config)

    return config


def readJsonc(jsoncFileName: str):
    """
        读取jsonc文件并去除注释
    """
    if not os.path.exists(jsoncFileName):
        raise Exception(f"{jsoncFileName} not found")
    with open(jsoncFileName, "r") as f:
        content = f.read()

    return removeComments(content)


def removeComments(json_string):
    comment_regex = re.compile(r"(?m)(?s)//.*?$|/\*.*?\*/")
    """
        去除jsonc文件中的注释
    """
    tmp = comment_regex.sub("", json_string)
    whitespace_regex = re.compile(r"(?m)^\s*$[\r\n]*", re.MULTILINE)

    return whitespace_regex.sub("", tmp)


def to_namespace(data):
    """
        将嵌套字典转成namespace
    """
    if not isinstance(data, dict):
        return data

    namespace = SimpleNamespace()
    for k, v in data.items():
        namespace.__setattr__(k, to_namespace(v))

    return namespace


def get(url):
    r = requests.get(url)
    try:
        r.raise_for_status()
    except:
        print("error", r.text, r.status_code)
        return None, r.status_code
    else:
        return r.content, r.status_code


def post(url, data):
    r = requests.post(url, data=data)
    try:
        r.raise_for_status()
    except:
        print("error", r.text, r.status_code)
        return None, r.status_code
    else:
        return r.content, r.status_code


class RPCClient:
    """
    用python调用RPC server(go)提供的HTTP服务
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        # 加载配置文件
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        config = loadConfig(os.path.join(self.current_path, "rpc_server", "config.json"))
        self.http_port = config.port.HttpPort
        self.storage_method = config.storage.Method
        self.save_dir = config.model.ModelPath
        # 以子进程的形式启动RPC server（go）
        self.start_rpc_server()

        if self.storage_method == "memory":
            raise NotImplementedError
        elif self.storage_method == "tmpfs":
            pass
        elif self.storage_method == "disk":
            raise NotImplementedError
        else:
            raise ValueError

    def __del__(self):
        # 需要优雅的结束RPC server以及torrent.Client
        self.stop_rpc_server()

    def start_rpc_server(self):
        rpc_server_thread = Thread(target=self.rpc_server)
        # 主线程退出时子线程退出
        rpc_server_thread.setDaemon(True)
        rpc_server_thread.start()
        # 子进程启动后并不意味着RPC服务就已经启动（需要启动时间）
        # TODO:在使用RPC服务之前需要进行进程同步
        sleep(1)

    def stop_rpc_server(self):
        # It's not enough to only kill the subprocess, as its subprocesses that listen to the dataport or httpport are still alive, which becode the zombie processes.
        # self.rpc_server_process.terminate()
        # This method works well in most of the cases, but the return value is not 0 but 1.
        os.killpg(os.getpgid(self.rpc_server_process.pid), signal.SIGKILL)
        # This method doesn't work sometimes.
        # parent_proc = psutil.Process(self.rpc_server_process.pid)
        # for child_proc in parent_proc.children(recursive=True):
        #     child_proc.kill()
        # parent_proc.kill()

    def rpc_server(self):
        log_path = os.path.join(self.current_path, f"rpc_server_{dist.get_rank()}.log")
        bin_path = os.path.join(self.current_path, "rpc_server", "rpc_server.bin")
        config_path = os.path.join(self.current_path, "rpc_server", "config.json")
        cmd = f"{bin_path} -config {config_path}"
        self.logger.info(f"start rpc server cmd: {cmd}")

        with open(log_path, "wb") as f:
            self.rpc_server_process = Popen(
                cmd,
                shell=True,
                stdout=f,
                stderr=f,
            )

    def create_torrent(self, path: str):
        """
        create torrent.

        Args:
            path (str): path of the file to be seeding

        Returns:
            torrent (bytes): metainfo of file
            status (bool): status_code == 200 OR NOT
        """
        url = f"http://localhost:{self.http_port}/create_torrent/"
        if self.storage_method == "memory":
            data = {
                "mb": {
                    "Data": [],  # bytes
                    "Length": 1,
                },
                "path": None,
            }
            raise NotImplementedError
        elif self.storage_method == "tmpfs":
            data = {
                "mb": None,
                "path": path,
            }
        elif self.storage_method == "disk":
            data = {
                "mb": None,
                "path": path,
            }
            raise NotImplementedError
        else:
            raise ValueError

        torrent, status_code = post(url, json.dumps(data))
        return torrent, status_code == requests.codes.ok

    def start_seeding(self, torrent):
        """
        start seeding.

        Args:
            torrent (bytes): metainfo of file to be seeding

        Returns:
            text (str): None if start seeding successes. Else, error message.
            status (bool): status_code == 200 OR NOT
        """
        url = f"http://localhost:{self.http_port}/start_seeding/"
        text, status_code = post(url, torrent)
        return (
            text.decode(encoding="utf-8", errors="ignore"),
            status_code == requests.codes.ok,
        )

    def stop_seeding(self, torrent):
        """
        stop seeding.

        Args:
            torrent (bytes): metainfo of the file

        Returns:
            text (str): None if start seeding successes. Else, error message.
            status (bool): status_code == 200 OR NOT
        """
        url = f"http://localhost:{self.http_port}/stop_seeding/"
        text, status_code = post(url, torrent)
        return text.decode(encoding="utf-8", errors="ignore"), status_code == 200

    def get_torrent_status(self, torrent):
        """
        get torrent status.

        Args:
            torrent (bytes): metainfo of the file

        Returns:
            torrent_status(dict): {exist, seeding}
            status (bool): status_code == 200 OR NOT
        """
        url = f"http://localhost:{self.http_port}/get_torrent_status/"
        torrent_status, status_code = post(url, torrent)
        return json.loads(torrent_status), status_code == 200

    def start_downloading(self, torrent):
        """
        download the file specified in torrent to model.ModelPath configured in config.jsonc.

        Args:
            torrent (bytes): metainfo of the file
            dst_path (str): destination path

        Returns:
            torrent_status(dict): {exist, seeding}
            status (bool): status_code == 200 OR NOT
        """
        url = f"http://localhost:{self.http_port}/start_downloading/"
        downloading_output, status_code = post(url, torrent)
        return json.loads(downloading_output), status_code == 200


if __name__ == "__main__":
    rpc_client = RPCClient()

    torrent, status = rpc_client.create_torrent("/dev/shm/bert_base_model.pth")
    print(torrent)
    if not status:
        print("create torrent failed")
        exit(0)

    torrent_status, status = rpc_client.get_torrent_status(torrent)
    print(torrent_status)
    if not status:
        print("get torrent status failed")
        exit(0)

    text, status = rpc_client.start_seeding(torrent)
    print(text)
    if not status:
        print("start seeding failed")
        exit(0)

    torrent_status, status = rpc_client.get_torrent_status(torrent)
    print(torrent_status)
    if not status:
        print("get torrent status failed")
        exit(0)

    text, status = rpc_client.stop_seeding(torrent)
    print(text)
    if not status:
        print("stop seeding failed")
        exit(0)

    torrent_status, status = rpc_client.get_torrent_status(torrent)
    print(torrent_status)
    if not status:
        print("get torrent status failed")
        exit(0)

    with open("./torrent/music_40MB.torrent", "rb") as f:
        torrent = f.read()
    downloading_output, status = rpc_client.start_downloading(torrent)
    print(downloading_output)
    if not status:
        print("download failed")
        exit(0)
