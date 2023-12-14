import pandas as pd
import re


def func():
    file_name = "./rpc_server_1.log"
    # 根据IP找端口
    # pattern = r'192.168.124.104:(\d+)'
    # result = set()

    # 192.168.124.104:59934 piece bitmap:
    # 找参与rarity计算的连接
    pattern = ": (.*?):(\d+) piece bitmap:"
    result = set()

    # request chunk indexes: []
    # 找request indexes不为空的
    # pattern = r"request chunk indexes: \[(\d+)"
    # result = set()

    with open(file_name, 'r') as f:
        # w = open("parsed_rpc_server_1.log",'w')
        for line in f:
            match = re.search(pattern, line)
            if match:
                result.add(":".join(match.groups()))
                # w.write(line+"\n")
        # w.close()
    print(sorted(result))


def distribution_aggregation():
    # 正则表达式来匹配日志行
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[\d\] ".+", line \d+, INFO: \[\d \d+\] aggregation:(\d+\.\d+) distribution:(\d+\.\d+)'

    # 用于存储解析数据的列表
    data = []

    # 读取日志文件
    with open('./distribution_aggregation.txt', 'r') as file:
        for line in file:
            match = re.search(log_pattern, line)
            if match:
                timestamp, aggregation, distribution = match.groups()
                data.append([timestamp, float(aggregation), float(distribution)])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'aggregation', 'distribution'])

    # 导出为CSV
    df.to_csv('distribution_aggregation.csv', index=False)

    print("Data exported to 'output.csv'")


if __name__ == "__main__":
    distribution_aggregation()
