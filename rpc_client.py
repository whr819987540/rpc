"""
    RPCClient调用go提供的数据传输服务
    TorrentCommunication继续封装RPCClient的功能, 形成三个原语
"""

import os
import re
import json
from types import SimpleNamespace
from subprocess import Popen
from threading import Thread
import requests
from time import sleep
import signal
from torch import distributed as dist
import torch
import numpy as np
from typing import Union
import logging
from concurrent.futures import ThreadPoolExecutor


class TorrentCommunication:
    def __init__(self, rank:int, logger:logging.Logger):
        """
            实例化RPCClient
        """
        self.rank = rank
        self.rpc_client = RPCClient(logger)
        self.logger = logger
        self.thread_pool = ThreadPoolExecutor(2)

    def bt_broadcast(self):
        """
            For server, data_path is str and this function returns torrent. If this function failed, an Exception will be raised.

            For client, data_path is None and this function returns torrent.

            This function needs to be rewritten according to the specific communication backend.
        """
        pass

    def bt_recv(self, torrent: bytes):
        # TODO: 这个函数可以是非阻塞的
        # DONE: 这个函数没必要是非阻塞的
        # 因为bt-ps以所有梯度为单位进行分发, 目前没有实现细粒度的通信, 通信与计算在这里无法并行, 也就没必要改成非阻塞
        downloading_output, status = self.rpc_client.start_downloading(torrent)
        if not status:
            raise Exception("bt_recv downloading error")
        else:
            return downloading_output

    def stop_seeding(self, torrent: bytes):
        text, status = self.rpc_client.stop_seeding(torrent)
        if not status:
            raise Exception(text)

    @staticmethod
    def _start_seeding(self, torrent:bytes):
        self.logger.debug("_start_seeding")
        text, status = self.rpc_client.start_seeding(torrent)
        if not status:
            return Exception(text)
        self.logger.debug("_start_seeding is done")

    @staticmethod
    def _broadcast_torrent(self, torrent:bytes):
        """
            Broadcast torrent to all clients.

            This function needs to be rewritten according to the specific communication backend.
        """
        pass


class TorrentCommunicationPyTorch:
    """
        封装RPCClient(python)提供的服务
        只提供两个通信原语与一个停止上传原语
        通信原语1: 
            如果是server, bt_broadcast将数据广播给所有client 
                1) 生成torrent
                2) 对数据进行seed
                2) 将torrent广播(这个广播操作借用了pytorch.distributed提供的broadcast)给所有client, client收到后会以torrent为参数通过bt_recv操作开始下载
            如果是client, bt_broadcast负责接收torrent
            无论是server还是client, 这个函数都会返回torrent

        通信原语2: bt_recv通过bt进行下载
            1) 以torrent为参数调用RPCClient.start_downloading
            2) 再调用RPCServer.start_downloading

        停止上传原语: server/client在完成下载后, 默认都会继续上传, 该原语终止上传行为
    """

    def __init__(self, logger: logging.Logger):
        """
            实例化RPCClient
        """
        self.rpc_client = RPCClient(logger)
        self.logger = logger
        self.thread_pool = ThreadPoolExecutor(2)

    def bt_broadcast(self, data_path: Union[str, None]):
        """
            For server, data_path is str and this function returns torrent. If this function failed, an Exception will be raised.

            For client, data_path is None and this function returns torrent.
        """
        rank = dist.get_rank()
        self.logger.debug(f"execute bt_broadcast: {data_path}")
        if rank == 0:
            # create torrent
            torrent, status = self.rpc_client.create_torrent(data_path)
            if not status:
                raise Exception("create torrent error")
            self.logger.debug("create torrent ok")

            # TODO: run seed and send torrent in parallel
            # DONE: 简单的进行并行通信即可, 如果server向tracker seed先完成, client后收到torrent, 当然是可以的
            # 在现在的实现下需要RTT(server,tracker)<2*RTT(server,client), 总时间为2*RTT(server,client)+RTT(client,tracker)
            # 如果client先收到torrent, 之后server才成功地向tracker seed,即RTT(server,tracker)>2*RTT(server,client), 
            # 总时间将变为RTT(server,tracker)+RTT(client,tracker)+interval
            # 目前这种情况不会发生, 因为server到client需要两次broadcast, 且server到tracker的通信是很快的
            # seed
            future_list = []
            future_list.append(self.thread_pool.submit(self._start_seeding, self, torrent))
            # text, status = self.rpc_client.start_seeding(torrent)
            # if not status:
            #     raise Exception(text)

            # send torrent
            future_list.append(self.thread_pool.submit(self._broadcast_torrent, self, torrent))
            # torrent_tensor = torch.from_numpy(np.frombuffer(torrent, np.uint8))
            # # TODO: 这里存在两个发送，增加了时延，通过别的发送方式可以优化
            # # size
            # dist.broadcast(torch.tensor([torrent_tensor.shape[0]], dtype=torch.int64), 0)
            # # data
            # dist.broadcast(torrent_tensor, 0)

            for future in future_list:
                result = future.result()
                if result is None:
                    continue
                else:
                    raise result
        else:
            # size
            self.logger.debug("clinet bt_broadcast")
            torrent_size = torch.empty(1, dtype=torch.int64)
            dist.broadcast(torrent_size, 0)
            # data
            torrent_tensor = torch.empty(torrent_size[0], dtype=torch.uint8)
            dist.broadcast(torrent_tensor, 0)
            torrent = torrent_tensor.numpy().tobytes()
            self.logger.debug("clinet bt_broadcast is done")

        return torrent

    def bt_recv(self, torrent: bytes):
        # TODO: 这个函数可以是非阻塞的
        # DONE: 这个函数没必要是非阻塞的
        # 因为bt-ps以所有梯度为单位进行分发, 目前没有实现细粒度的通信, 通信与计算在这里无法并行, 也就没必要改成非阻塞
        downloading_output, status = self.rpc_client.start_downloading(torrent)
        if not status:
            raise Exception("bt_recv downloading error")
        else:
            return downloading_output

    def stop_seeding(self, torrent: bytes):
        text, status = self.rpc_client.stop_seeding(torrent)
        if not status:
            raise Exception(text)

    @staticmethod
    def _start_seeding(self, torrent):
        self.logger.debug("_start_seeding")
        text, status = self.rpc_client.start_seeding(torrent)
        if not status:
            return Exception(text)
        self.logger.debug("_start_seeding is done")
        

    @staticmethod
    def _broadcast_torrent(self, torrent):
        self.logger.debug("_broadcast_torrent")
        torrent_tensor = torch.from_numpy(np.frombuffer(torrent, np.uint8))
        # TODO: 这里存在两个发送，增加了时延，通过别的发送方式可以优化
        # size
        dist.broadcast(torch.tensor([torrent_tensor.shape[0]], dtype=torch.int64), 0)
        # data
        dist.broadcast(torrent_tensor, 0)
        self.logger.debug("_broadcast_torrent is done")


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
    """
        去除jsonc文件中的注释
    """
    comment_regex = re.compile(r"(?m)(?s)// .*?$|/\*.*?\*/")
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

    def __init__(self, rank:int, logger: logging.Logger) -> None:
        self.rank = rank
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
        # DONE:最简单的方法是使用一下RPC的某个简单服务, 比如echo
        data = "hello"
        # sleep(3)
        redo_times = 3
        for i in range(redo_times):
            echo_data,status,e  = self.echo(data.encode())
            # RPC服务尚未启动
            if not status:
                self.logger.error(e)
                sleep(3)
                continue
            
            echo_data = echo_data.decode()
            self.logger.info(echo_data)
            # RPC服务出错
            if data != echo_data:
                self.logger.error(f"data: {data}, echo_data: {echo_data}")
                raise ValueError
            else:
                self.logger.info(f"really start rpc_server as a subprocess")
                break
        # 可能是因为达到最大的重做次数
        if not status:
            raise Exception(f"reach max redo times, failed to start RPC service")

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
        cmd = f"{bin_path} -config {config_path} -random_seed {self.rank}"
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

    def get_name(self, torrent):
        """
        get the name of the file or directory specified in torrent

        Args:
            torrent (bytes): metainfo of the file

        Returns:
            name(str): 
            status (bool): status_code == 200 OR NOT
        """
        url = f"http://localhost:{self.http_port}/get_name/"
        name, status_code = post(url, torrent)
        return name.decode(), status_code == 200

    def echo(self,data:Union[str,bytes],timeout:float=3.0):
        """
        echo something to server which will return exactly the same thing

        Args:
            data (str|bytes): 
            timeout (float): maximum waiting time for connect and response, 
                namely the maxinum waiting time for really starting the subprocess

        Returns:
            data(str|bytes):
            status (bool): status_code == 200 OR NOT
            exception
        """
        if type(data) is str:
            data = data.encode()
        elif type(data) is bytes:
            pass
        else:
            raise TypeError

        url = f"http://localhost:{self.http_port}/echo/"
        try:
            # :param timeout: (optional) How many seconds to wait for the server to send data
            # before giving up, as a float, or a :ref:`(connect timeout, read
            # timeout) <timeouts>` tuple.
            # :type timeout: float or tuple
            r = requests.post(url,data,timeout=(timeout,timeout))
        except Exception as e:
            return None, False, e

        try:
            r.raise_for_status()
        except Exception as e:
            return None, r.status_code == 200, e
        else:
            return r.content, r.status_code == 200, None

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
    model_name, status = rpc_client.get_name(torrent)
    print(downloading_output, rpc_client.save_dir, model_name.decode("utf-8"))
    if not status:
        print("download failed")
        exit(0)
