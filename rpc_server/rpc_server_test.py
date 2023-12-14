import requests

import re
import json


def load_config(path):
    """
    加载jsonc或json配置文件
    """
    with open(path) as f:
        config = f.read()
    config = remove_comments(config)
    return json.loads(config)


def remove_comments(text):
    # 匹配 // 注释并替换为空字符串
    pattern = r"// .*"
    result = re.sub(pattern, "", text)
    return result


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


# pass
def test_create_torret():
    """
    测试create_torrent
    """
    # tmpfs
    url = f"http://localhost:{httpPort}/create_torrent/"
    #  type createTorrentInput struct {
    # 	mb   storage.MemoryBuf
    # 	path string
    # }
    # data = {
    #         "mb": {
    #             "Data": None,
    #             "Length": 1
    #         },
    #         "path": "/dev/shm/bert_base_model.pth"
    #     }

    data = {"mb": None, "path": "/dev/shm/bert_base_model.pth"}

    ret = post(url, json.dumps(data))
    print(ret)

    ret = get(url)
    print(ret)


# pass
def test_start_seeding():
    # create_torrent
    url = f"http://localhost:{httpPort}/create_torrent/"
    data = {"mb": {"Data": None, "Length": 1}, "path": "/dev/shm/bert_base_model.pth"}
    torrent, status_code = post(url, json.dumps(data))
    if status_code != requests.codes.ok:
        print("create torrent failed")
        return
    else:
        print(torrent)

    # start_seeding
    url = f"http://localhost:{httpPort}/start_seeding/"
    ret = post(url, torrent)
    print(ret)


# pass
def test_stop_seeding():
    # create_torrent
    url = f"http://localhost:{httpPort}/create_torrent/"
    data = {"mb": {"Data": None, "Length": 1}, "path": "/dev/shm/bert_base_model.pth"}
    ret = post(url, json.dumps(data))
    if ret == None:
        print("create torrent failed")
        return
    else:
        print(ret)

    # stop_seeding
    url = f"http://localhost:{httpPort}/stop_seeding/"
    ret = post(url, ret)


# pass
def test_get_torrent_status():
    # create_torrent
    url = f"http://localhost:{httpPort}/create_torrent/"
    data = {"mb": {"Data": None, "Length": 1}, "path": "/dev/shm/bert_base_model.pth"}
    ret = post(url, json.dumps(data))
    if ret == None:
        print("create torrent failed")
        return
    else:
        print(ret)

    # get_torrent_status
    url = f"http://localhost:{httpPort}/get_torrent_status/"
    ret = post(url, ret)
    print(json.loads(ret))


def test_start_downloading():
    # load_torrent
    with open("./torrent/bert_base_model.pth.torrent", "rb") as f:
        torrent = f.read()

    # start_downloading
    url = f"http://localhost:{httpPort}/start_downloading/"
    downloading_output, status_code = post(url, torrent)
    print(json.loads(downloading_output), status_code)


def test_echo(timeout: float = 5.0):
    data = "hello".encode()
    url = f"http://localhost:{httpPort}/echo/"
    try:
        r = requests.post(url, data, timeout=(timeout, timeout))
    except Exception as e:
        print(e)
    else:
        print(r)


if __name__ == "__main__":
    # 加载配置文件
    config = load_config("./config.json")
    httpPort = config["port"]["HttpPort"]

    # test_create_torret()
    # test_start_seeding()
    # test_stop_seeding()
    # test_get_torrent_status()
    test_start_downloading()