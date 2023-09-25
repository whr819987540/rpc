package main

import (
	"fmt"
	"os"
)

// go读取二进制文件
func readModelParam(filePath string) (data []byte, err error) {
	fileSize, err := getFileSize(filePath)
	if err != nil {
		return
	}
	// 未检查堆区的闲置内存
	data = make([]byte, fileSize)
	file, err := os.Open(filePath)
	n, err := file.Read(data)
	if err != nil {
		return
	}
	// 这里可能有问题，n是int，最大只能到2G
	if n != int(fileSize) {
		err = fmt.Errorf("没有读完")
	}
	return
}

func getFileSize(filePath string) (fileSize int64, err error) {
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return
	}
	fileSize = fileInfo.Size()
	return
}
