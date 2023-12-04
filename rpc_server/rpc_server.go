package main

import (
	"bytes"
	"context"
	"encoding/json"
	"expvar"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"utils"

	"github.com/anacrolix/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/config"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
	"github.com/davecgh/go-spew/spew"
	"github.com/dustin/go-humanize"
)

var debugFlag bool
var data []byte
var startTime time.Time
var sendTimes int // 完成发送的次数
var recvTimes int // 完成接收的次数
var mutex sync.Mutex
var torrentURL string
var mi *metainfo.MetaInfo
var configStruct *config.Config
var storageMethod string          // 存储方法
var torrentClient *torrent.Client // 管理所有torrent的client
// Trackers are the urls of trackers
var Trackers [][]string

func accessLog(r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
}

func handleStatus(w http.ResponseWriter, r *http.Request) {
	accessLog(r)
	w.Write([]byte("hello"))
}

// client向server请求发送数据
func handleSend(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	var err error
	// 从收到第一个send开始计时
	if startTime.IsZero() {
		startTime = time.Now()
		log.Print("start timer")
	}

	// 还没有生成.torrent
	// create torrent from memory and seed
	if mi == nil {
		// memory, tmpfs, disk
		method := strings.ToLower(configStruct.Storage.Method)
		if method == "memory" {
			mi, err = fromMemory(data)
			info, err := infoBytesToInfo(mi.InfoBytes)
			if err != nil {
				log.Printf("infoBytesToInfo: %v", err)
			}
			totalLength := info.TotalLength()
			log.Printf("TotalLength %d\n", totalLength)

			mb := &storage.MemoryBuf{
				Data:   data,
				Length: totalLength,
			}
			seed(mi, mb)
		} else if method == "tmpfs" {
			modleParamPath := path.Join(configStruct.Model.ModelPath, configStruct.Model.ModelName)
			mi, err = fromTMPFS(modleParamPath) // 修改全局变量mi
			if err != nil {
				log.Printf("fromTMPFSFilePath: %v", err)
			}
			log.Printf("build MetaInfo and set all the fields")
			pprintMetainfo(mi, pprintMetainfoFlags{
				JustName:    false,
				PieceHashes: false,
				Files:       false,
			})

			info, err := mi.UnmarshalInfo()
			if err != nil {
				log.Printf("metainfo UnmarshalInfo: %v", err)
			} else {
				log.Printf("info: %v", info.Describe())
			}

			err = seedFromTMPFS(mi)
			if err != nil {
				log.Printf("seedFromTMPFS: %v", err)
			} else {
				log.Printf("seedFromTMPFS ok")
			}
		} else if method == "disk" {

		} else {

		}
	}

	err = mi.Write(w)
	if err != nil {
		log.Printf("send .torrent to %s error:%v", r.RemoteAddr, err)
		return
	}
	log.Printf("send .torrent to %s ok", r.RemoteAddr)
}

// client向server回传数据
func handleRecv(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	data, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}
	fmt.Fprintf(w, "OK") // POST请求需要有回复
	log.Printf("recv %d bytes from %s", len(data), r.RemoteAddr)

	mutex.Lock()
	defer mutex.Unlock()
	recvTimes++
	if recvTimes == configStruct.Client.TotalPeers {
		endTime := time.Now()
		log.Printf("PS, %s, total time: %v", configStruct.Model.ModelName, endTime.Sub(startTime))
	}
}

// client完成接收后, 通知server
func handleCompleteSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	data, err := io.ReadAll(r.Body)
	defer r.Body.Close()
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}
	w.Write([]byte("ok")) // POST请求需要有回复

	log.Printf("recv send status from %s,%s", r.RemoteAddr, string(data))
	mutex.Lock()
	defer mutex.Unlock()
	sendTimes++
	log.Printf("%s add sendTimes to %d", r.RemoteAddr, sendTimes)
}

// 获取完成向client发送的次数
func handleGetSendTimes(w http.ResponseWriter, r *http.Request) {
	mutex.Lock()
	defer mutex.Unlock()
	w.Write([]byte(strconv.Itoa(sendTimes)))
}

// 制作torrent文件
// - 名称：create_torrent
// - 输入：数据
//   - memory：pointer
//   - tmpfs：path
//   - disk：path
// - 方法：POST
// - 输出：torrent

// if stored in memory, data is not None
// if stored in tmpfs or disk, path is not None
type createTorrentInput struct {
	Mb   storage.MemoryBuf `json:"mb"`
	Path string            `json:"path"`
}

func create_torrent(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	dataBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("create_torrent read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("create_torrent read data ok")

	// convert json string to struct
	var input createTorrentInput
	err = json.Unmarshal(dataBytes, &input)
	if err != nil {
		log.Printf("create_torrent json unmarshal error: %v", err)
		http.Error(w, "Data malformat", http.StatusInternalServerError)
		return
	}
	log.Printf("create_torrent json unmarshal ok: %v", input)

	if storageMethod == "memory" {

	} else if storageMethod == "tmpfs" {
		mip, err := fromTMPFS(input.Path)
		if err != nil {
			log.Printf("create_torrent from tmpfs path error: %v", err)
			http.Error(w, fmt.Sprintf("create_torrent from tmpfs path error: %v", err), http.StatusInternalServerError)
			return
		}
		// 返回torrent
		err = mip.Write(w)
		if err != nil {
			log.Printf("return torrent to %s error: %v", r.RemoteAddr, err)
			return
		}
		log.Printf("create_torrent return torrent ok")
	} else if storageMethod == "disk" {

	} else {

	}
}

// 做种/上传

// - 名称：start_seeding
// - 输入：torrent
// - 输出：是否成功

func start_seeding(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	metaInfoBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("start_seeding read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("start_seeding read data ok")

	// MetaInfo
	var mi metainfo.MetaInfo
	d := bencode.NewDecoder(bytes.NewBuffer(metaInfoBytes))
	err = d.Decode(&mi)
	if err != nil {
		log.Printf("start_seeding bdecode torrent error: %v", err)
		http.Error(w, "Bdecode torrent failed", http.StatusInternalServerError)
	}
	log.Printf("start_seeding bdecode torrent ok")

	// seeding
	if storageMethod == "memory" {

	} else if storageMethod == "tmpfs" {
		err = seedFromTMPFS(&mi)
		if err != nil {
			log.Printf("seedFromTMPFS error: %v", err)
			http.Error(w, fmt.Sprintf("seedFromTMPFS error: %v", err), http.StatusInternalServerError)
			return
		}
		log.Printf("seedFromTMPFS ok")
	} else if storageMethod == "disk" {

	} else {

	}
}

// - 停止做种
//   - 名称：stop_seeding
//   - 输入：torrent
//   - client与torrent
//   - memory：一对一，需要一个专门的管理器
//   - tmpfs：一对所有，直接卸载相应的torrent
//   - disk：一对所有，直接卸载相应的torrent
//   - 方法：POST
//   - 输出：是否成功

func stop_seeding(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	metaInfoBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("stop_seeding read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("stop_seeding read data ok")

	// MetaInfo
	var mi metainfo.MetaInfo
	d := bencode.NewDecoder(bytes.NewBuffer(metaInfoBytes))
	err = d.Decode(&mi)
	if err != nil {
		log.Printf("stop_seeding bdecode torrent error: %v", err)
		http.Error(w, "Bdecode torrent failed", http.StatusInternalServerError)
	}
	log.Printf("stop_seeding bdecode torrent ok")

	// stop seeding
	if storageMethod == "memory" {

	} else if storageMethod == "tmpfs" {
		hib := mi.HashInfoBytes()
		t, ok := torrentClient.Torrent(hib)
		// if the torrent doesn't exist, return 200 is ok
		// cause we have nothing to stop
		if !ok {
			log.Printf("stop_seeding finds the torrent not in the client's torrent list: %s", mi.Describe())
			return
		}
		// Drop the torrent from the client, and close it.
		// No methods like pause are provided.
		t.Drop()

	} else if storageMethod == "disk" {

	} else {

	}
}

// 检查种子的状态

// - 名称：get_torrent_status
// - 输入：torrent
// - 输出：状态
//   - 被加入client
//     - 是否seeding
//   - 未被加入client

type getTorrentStatusOutput struct {
	Exist   bool `json:"exist"`
	Seeding bool `json:"seeding"`
}

func get_torrent_status(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	metaInfoBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("get_torrent_status read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("get_torrent_status read data ok")

	// MetaInfo
	var mi metainfo.MetaInfo
	d := bencode.NewDecoder(bytes.NewBuffer(metaInfoBytes))
	err = d.Decode(&mi)
	if err != nil {
		log.Printf("get_torrent_status bdecode torrent error: %v", err)
		http.Error(w, "Bdecode torrent failed", http.StatusInternalServerError)
		return
	}
	log.Printf("get_torrent_status bdecode torrent ok")

	// get status
	var status getTorrentStatusOutput
	if storageMethod == "memory" {

	} else if storageMethod == "tmpfs" {
		// exist
		hib := mi.HashInfoBytes()
		t, ok := torrentClient.Torrent(hib)
		status.Exist = ok
		if ok {
			status.Seeding = t.Seeding()
		} else {
			status.Seeding = false
		}
	} else if storageMethod == "disk" {

	} else {

	}

	// return status
	statusJson, err := json.Marshal(status)
	if err != nil {
		log.Printf("get_torrent_status json marshal error: %v", err)
		http.Error(w, "Json marshal torrent status failed", http.StatusInternalServerError)
		return
	}
	n, err := w.Write(statusJson)
	if err != nil {
		log.Printf("get_torrent_status write status to %s error: %v", r.RemoteAddr, err)
		return
	}
	log.Printf("get_torrent_status write %d bytes status to %s ok", n, r.RemoteAddr)
}

// 下载

// - 名称：start_downloading
// - 输入：torrent
// - 方法：POST
// - 输出：下载文件的位置
//   - memory：
//   - tmpfs：下载位置
//   - disk：下载位置

type startDownloadingOutput struct {
	createTorrentInput
}

func start_downloading(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	metaInfoBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("start_downloading read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("start_downloading read data ok")

	// MetaInfo
	var mi metainfo.MetaInfo
	d := bencode.NewDecoder(bytes.NewBuffer(metaInfoBytes))
	err = d.Decode(&mi)
	if err != nil {
		log.Printf("start_downloading bdecode torrent error: %v", err)
		http.Error(w, "Bdecode torrent failed", http.StatusInternalServerError)
		return
	}
	log.Printf("start_downloading bdecode torrent ok")

	// Info
	info, err := mi.UnmarshalInfo()
	if err != nil {
		log.Printf("start_downloading unmarshal info bytes error: %v", err)
		http.Error(w, "Unmarshal info bytes failed", http.StatusInternalServerError)
		return
	}

	// 向client中添加torrent
	t, err := torrentClient.AddTorrent(&mi)
	if err != nil {
		log.Printf("start_downloading add torrent error: %v", err)
		http.Error(w, "start_downloading add torrent failed", http.StatusInternalServerError)
		return
	}

	// ctx will be cancelled when os.Interrupt signal is emitted
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	// create a goroutine to print the download process
	utils.TorrentBar(t, false)
	var wg sync.WaitGroup
	wg.Add(1)
	// create a goroutine to download
	go func() {
		defer wg.Done()
		select {
		case <-ctx.Done():
			return
		case <-t.GotInfo():
		}

		t.DownloadAll() // 只是声明哪些piece(所有)需要被下载
		wg.Add(1)
		go func() { // 用goroutine来检查所有的piece都已经被下载(发布/订阅模式)
			defer wg.Done()
			utils.WaitForPieces(ctx, t, 0, t.NumPieces())
		}()
	}()

	started := time.Now()
	defer utils.OutputStats(torrentClient)
	wg.Wait()

	if ctx.Err() == nil {
		log.Print("downloaded ALL the torrents")
	} else {
		err = ctx.Err()
	}
	clientConnStats := torrentClient.ConnStats()
	log.Printf("average download rate: %v",
		humanize.Bytes(
			uint64(
				time.Duration(
					clientConnStats.BytesReadUsefulData.Int64(),
				)*time.Second/time.Since(started),
			),
		),
	)

	spew.Dump(expvar.Get("torrent").(*expvar.Map).Get("chunks received"))
	spew.Dump(torrentClient.ConnStats())
	clStats := torrentClient.ConnStats()
	sentOverhead := clStats.BytesWritten.Int64() - clStats.BytesWrittenData.Int64()
	log.Printf(
		"client read %v, %.1f%% was useful data. sent %v non-data bytes",
		humanize.Bytes(uint64(clStats.BytesRead.Int64())),
		100*float64(clStats.BytesReadUsefulData.Int64())/float64(clStats.BytesRead.Int64()),
		humanize.Bytes(uint64(sentOverhead)),
	)

	var output startDownloadingOutput
	if storageMethod == "memory" {

	} else if storageMethod == "tmpfs" {
		output.Path = path.Join(configStruct.Model.ModelPath, info.BestName())
		outputJson, err := json.Marshal(output)
		if err != nil {
			log.Printf("start_downloading json marshal error: %v", err)
			http.Error(w, "Json marshal start_downloading output failed", http.StatusInternalServerError)
			return
		}
		n, err := w.Write(outputJson)
		if err != nil {
			log.Printf("start_downloading write output to %s error: %v", r.RemoteAddr, err)
			return
		}
		log.Printf("start_downloading write %d bytes output to %s ok", n, r.RemoteAddr)
	} else if storageMethod == "disk" {

	} else {

	}
}

func get_name(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	metaInfoBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("start_downloading read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("start_downloading read data ok")

	// MetaInfo
	var mi metainfo.MetaInfo
	d := bencode.NewDecoder(bytes.NewBuffer(metaInfoBytes))
	err = d.Decode(&mi)
	if err != nil {
		log.Printf("start_downloading bdecode torrent error: %v", err)
		http.Error(w, "Bdecode torrent failed", http.StatusInternalServerError)
		return
	}
	log.Printf("start_downloading bdecode torrent ok")

	// Info
	info, err := mi.UnmarshalInfo()
	if err != nil {
		log.Printf("start_downloading unmarshal info bytes error: %v", err)
		http.Error(w, "Unmarshal info bytes failed", http.StatusInternalServerError)
		return
	}

	n, err := w.Write(([]byte(info.BestName())))
	if err != nil {
		log.Printf("get_name write status to %s error: %v", r.RemoteAddr, err)
		return
	}
	log.Printf("get_name write %d bytes status to %s ok", n, r.RemoteAddr)
}

func echo(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if r.Method != "POST" {
		log.Printf("Invalid request method %s", r.Method)
		http.Error(w, fmt.Sprintf("Invalid request method %s", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// read data
	data, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("echo read data error: %v", err)
		http.Error(w, "Read data failed", http.StatusInternalServerError)
		return
	}
	log.Printf("echo read data ok")

	n, err := w.Write(data)
	if err != nil {
		log.Printf("echo send back to %s error: %v", r.RemoteAddr, err)
		return
	}
	log.Printf("echo send back %d bytes to %s ok", n, r.RemoteAddr)
}

func f(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s from %s", r.Method, r.RequestURI, r.RemoteAddr)
	if storageMethod == "memory" {

	} else if storageMethod == "tmpfs" {

	} else if storageMethod == "disk" {

	} else {

	}
}

func httpFunc() {
	http.HandleFunc("/status/", handleStatus)
	http.HandleFunc("/send/", handleSend)
	http.HandleFunc("/recv/", handleRecv)
	http.HandleFunc("/completesend/", handleCompleteSend)
	http.HandleFunc("/sendtimes/", handleGetSendTimes)

	http.HandleFunc("/create_torrent/", create_torrent)
	http.HandleFunc("/start_seeding/", start_seeding)
	http.HandleFunc("/stop_seeding/", stop_seeding)
	http.HandleFunc("/get_torrent_status/", get_torrent_status)
	http.HandleFunc("/start_downloading/", start_downloading)
	http.HandleFunc("/get_name/", get_name)
	http.HandleFunc("/echo/", echo)

	if err := http.ListenAndServe(fmt.Sprintf(":%d", configStruct.Port.HTTPPort), nil); err != nil {
		log.Printf("listen %d error", configStruct.Port.HTTPPort)
	}
}

func main() {
	var err error

	jsoncFileName := flag.String("config", "config.json", "set the config file path.")
	flag.Parse()

	// 加载配置数据
	configStruct, err = config.LoadJsonc(*jsoncFileName)
	if err != nil {
		log.Printf("load config error: %v", err)
		return
	}
	log.Printf("configStruct %v", configStruct)

	storageMethod = strings.ToLower(configStruct.Storage.Method)
	Trackers = configStruct.Tracker.URLList
	debugFlag = configStruct.TorrentLib.Debug

	// 设置torrent.Client
	// client config
	clientConfig := torrent.NewDefaultClientConfig()
	// 对于seeder, 一开始就上传
	// 对于leecher, 下载结束后也应该继续上传, 直到手动取消
	clientConfig.Seed = true
	// 监听哪个端口并接收peer的连接
	clientConfig.SetListenAddr(fmt.Sprintf(":%d", configStruct.Port.DataPort))
	// 默认开启TCP/UTP/IPV4/IPV6
	clientConfig.DisableAcceptRateLimiting = true
	clientConfig.PublicIp6 = nil // 必须设置为nil或设置为真实值, 不能为空, 否则utp会使用dht, 然后报错
	clientConfig.PublicIp4 = nil
	clientConfig.Debug = debugFlag
	if storageMethod == "memory" {
		// 如果直接存储在内存中, 一个torrent.Client只能管理一个torrent
	} else if storageMethod == "tmpfs" {
		// 指定torrent data的存储路径
		storageImplCloser := storage.NewFile(configStruct.Model.ModelPath)
		clientConfig.DefaultStorage = storageImplCloser

		// client
		torrentClient, err = torrent.NewClient(clientConfig)
		if err != nil {
			log.Printf("create torrent.Client for tmpfs error: %v", err)
			return
		}
		log.Printf("create torrent.Client for tmpfs")
		defer torrentClient.Close() // exit elegantly
	} else if storageMethod == "disk" {

	} else {

	}

	clientConfigString := ""
	v := reflect.ValueOf(clientConfig).Elem()
	for i := 0; i < v.NumField(); i++ {
		k := v.Type().Field(i).Name
		v := v.Field(i).Interface()
		clientConfigString += fmt.Sprintf("%v: %v", k, v) + "\n"
	}
	log.Printf("info: clientConfig %s", clientConfigString)

	// 启动
	httpFunc()
}
