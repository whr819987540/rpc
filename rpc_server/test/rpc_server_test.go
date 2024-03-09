package test

import (
	"bytes"
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/netip"
	"os"
	"os/signal"
	"reflect"
	"sort"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
	"utils"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/config"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
	"github.com/dustin/go-humanize"
)

func TestPiecePossession(t *testing.T) {
	torrentPath := "../torrent/music_40MB.torrent"
	// /home/whr/code/communication/bt_ps/p2p_server/rpc/rpc_server/rpc_server.bin -config /home/whr/code/communication/bt_ps/p2p_server/rpc/rpc_server/config.json
	metaInfoBytes, err := os.ReadFile(torrentPath)
	if err != nil {
		t.Errorf("read file error: %v", err)
	}
	// fmt.Println(torrent)
	download(t, metaInfoBytes)
}

func download(t *testing.T, metaInfoBytes []byte) {
	var err error
	// MetaInfo
	var mi metainfo.MetaInfo
	d := bencode.NewDecoder(bytes.NewBuffer(metaInfoBytes))
	err = d.Decode(&mi)
	if err != nil {
		t.Errorf("start_downloading bdecode torrent error: %v", err)
		return
	}
	t.Log("start_downloading bdecode torrent ok")

	// // Info
	// info, err := mi.UnmarshalInfo()
	// if err != nil {
	// 	t.Errorf("start_downloading unmarshal info bytes error: %v", err)
	// 	return
	// }
	// t.Log(info)

	// client
	var torrentClient *torrent.Client // 管理所有torrent的client
	var configStruct *config.Config
	var storageMethod string // 存储方法

	// 加载配置数据
	configStruct, err = config.LoadJsonc("/home/whr/code/communication/bt_ps/p2p_server/rpc/rpc_server/config.json")
	if err != nil {
		t.Logf("load config error: %v", err)
		return
	}
	t.Logf("configStruct %v", configStruct)

	storageMethod = strings.ToLower(configStruct.Storage.Method)

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
	clientConfig.Debug = true

	clientConfigString := ""
	v := reflect.ValueOf(clientConfig).Elem()
	for i := 0; i < v.NumField(); i++ {
		k := v.Type().Field(i).Name
		v := v.Field(i).Interface()
		clientConfigString += fmt.Sprintf("%v: %v", k, v) + "\n"
	}
	t.Logf("info: clientConfig %s", clientConfigString)

	if storageMethod == "memory" {
		// 如果直接存储在内存中, 一个torrent.Client只能管理一个torrent
	} else if storageMethod == "tmpfs" {
		// 指定torrent data的存储路径
		storageImplCloser := storage.NewFile(configStruct.Model.ModelPath)
		clientConfig.DefaultStorage = storageImplCloser

		// client
		torrentClient, err = torrent.NewClient(clientConfig)
		if err != nil {
			t.Logf("create torrent.Client for tmpfs error: %v", err)
			return
		}
		t.Logf("create torrent.Client for tmpfs")
		defer torrentClient.Close() // exit elegantly
	} else if storageMethod == "disk" {

	} else {

	}

	// 向client中添加torrent
	torrent, err := torrentClient.AddTorrent(&mi)
	if err != nil {
		t.Errorf("start_downloading add torrent error: %v", err)
		return
	}

	// ctx will be cancelled when os.Interrupt signal is emitted
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	// create a goroutine to print the download process
	utils.TorrentBar(torrent, false)
	var wg sync.WaitGroup
	wg.Add(1)
	// create a goroutine to download
	go func() {
		defer wg.Done()
		select {
		case <-ctx.Done():
			return
		case <-torrent.GotInfo():
		}

		torrent.DownloadAll() // 只是声明哪些piece(所有)需要被下载
		wg.Add(1)
		go func() { // 用goroutine来检查所有的piece都已经被下载(发布/订阅模式)
			defer wg.Done()
			utils.WaitForPieces(ctx, torrent, 0, torrent.NumPieces())
		}()
	}()

	started := time.Now()
	defer utils.OutputStats(torrentClient)
	wg.Wait()

	if ctx.Err() == nil {
		t.Log("downloaded ALL the torrents")
	} else {
		err = ctx.Err()
	}
	clientConnStats := torrentClient.ConnStats()
	t.Logf("average download rate: %v",
		humanize.Bytes(
			uint64(
				time.Duration(
					clientConnStats.BytesReadUsefulData.Int64(),
				)*time.Second/time.Since(started),
			),
		),
	)

	// spew.Dump(expvar.Get("torrent").(*expvar.Map).Get("chunks received"))
	// spew.Dump(torrentClient.ConnStats())
	// clStats := torrentClient.ConnStats()
	// sentOverhead := clStats.BytesWritten.Int64() - clStats.BytesWrittenData.Int64()
	// log.Printf(
	// 	"client read %v, %.1f%% was useful data. sent %v non-data bytes",
	// 	humanize.Bytes(uint64(clStats.BytesRead.Int64())),
	// 	100*float64(clStats.BytesReadUsefulData.Int64())/float64(clStats.BytesRead.Int64()),
	// 	humanize.Bytes(uint64(sentOverhead)),
	// )

	// var output startDownloadingOutput
	// if storageMethod == "memory" {

	// } else if storageMethod == "tmpfs" {
	// 	output.Path = path.Join(configStruct.Model.ModelPath, info.BestName())
	// 	outputJson, err := json.Marshal(output)
	// 	if err != nil {
	// 		log.Printf("start_downloading json marshal error: %v", err)
	// 		http.Error(w, "Json marshal start_downloading output failed", http.StatusInternalServerError)
	// 		return
	// 	}
	// 	n, err := w.Write(outputJson)
	// 	if err != nil {
	// 		log.Printf("start_downloading write output to %s error: %v", r.RemoteAddr, err)
	// 		return
	// 	}
	// 	log.Printf("start_downloading write %d bytes output to %s ok", n, r.RemoteAddr)
	// } else if storageMethod == "disk" {

	// } else {

	// }
}

func TestLoadJsonc(t *testing.T) {
	jsoncFileName := "/home/whr/code/communication/bt_ps/p2p_server/rpc/rpc_server/config.json"
	configStruct, err := config.LoadJsonc(jsoncFileName)
	if err != nil {
		t.Errorf("load config error: %v", err)
		return
	}
	t.Logf("configStruct %v", configStruct)
}

func TestSortByValueWithIndex(t *testing.T) {
	arr := []int{5, 2, 8, 3, 1}
	// 获取降序排序后的索引
	sortedIndexes := sortByValueWithIndex(arr)
	// 打印排序后的索引
	t.Logf("%v", sortedIndexes)
}

func sortByValueWithIndex(arr []int) []int {
	// 创建一个结构体切片，用于保存值和对应的索引
	type kv struct {
		value, index int
	}

	// 初始化结构体切片
	arrWithIndex := make([]kv, len(arr))
	for i := range arr {
		arrWithIndex[i] = kv{arr[i], i}
	}

	// 使用sort.Slice函数进行降序排序，并指定比较函数
	sort.Slice(arrWithIndex, func(i, j int) bool {
		return arrWithIndex[i].value > arrWithIndex[j].value
	})

	// 创建一个切片用于保存排序后的索引
	sortedIndexes := make([]int, len(arrWithIndex))
	for i := range arrWithIndex {
		sortedIndexes[i] = arrWithIndex[i].index
	}

	return sortedIndexes
}

func TestSliceRange(t *testing.T) {
	array := []int{1, 2, 3}
	// return index
	for i := range array {
		t.Logf("%d", i)
	}
	// return index and value
	for k, v := range array {
		t.Logf("%d %d", k, v)
	}
}

type (
	RarityContentType = uint16
	RequestIndex      = uint32
)

func TestClassifyRarity(t *testing.T) {
	rarityArray := []RarityContentType{3, 2, 0, 1, 0, 3, 4, 2, 4}
	// category number
	categoryNumber := countUniqueElements(rarityArray)
	// classify rarity and record the index
	bucktes := make([][]RequestIndex, categoryNumber)
	for k, v := range rarityArray {
		bucktes[v] = append(bucktes[v], RequestIndex(k))
	}

	t.Logf("%v", bucktes)
	for rarity, row := range bucktes {
		t.Logf("ratiry %d: %v", rarity, row)
	}
	for rarity, row := range bucktes {
		for _, index := range row {
			if RarityContentType(rarity) != rarityArray[index] {
				t.Errorf("not equal.")
			}
		}
	}

	// furthermore, randomize the indexes
	for _, indexes := range bucktes {
		rand.Seed(0)
		rand.Shuffle(
			len(indexes),
			func(i, j int) {
				indexes[i], indexes[j] = indexes[j], indexes[i]
			},
		)
	}
	t.Logf("%v", bucktes)
	for rarity, row := range bucktes {
		t.Logf("ratiry %d: %v", rarity, row)
	}
}

func countUniqueElements(arr []RarityContentType) (uniqueCount int) {
	elementCount := make(map[RarityContentType]int)
	// 遍历数组，统计元素出现次数
	for _, element := range arr {
		elementCount[element] = 1
	}
	// 统计不同元素的种类数
	for range elementCount {
		uniqueCount++
	}
	return
}

func TestRand(t *testing.T) {
	rarityArray := []RarityContentType{3, 2, 0, 1, 0, 3, 4, 2, 4}
	t.Logf("%v", rarityArray)
	// this function wraps Fisher-Yates shuffle function
	rand.Seed(0)
	rand.Shuffle(
		len(rarityArray),
		func(i, j int) {
			rarityArray[i], rarityArray[j] = rarityArray[j], rarityArray[i]
		},
	)
	t.Logf("%v", rarityArray)

}

type PieceSelectionStrategyEnum int

const (
	RandomSelectionStrategy PieceSelectionStrategyEnum = 0
	RFSelectionStrategy     PieceSelectionStrategyEnum = 1
)

type testPieceSelection struct {
	PieceSelectionStrategy PieceSelectionStrategyEnum
}

func TestEnum(t *testing.T) {
	var tps testPieceSelection
	tps.PieceSelectionStrategy = RFSelectionStrategy
	t.Logf("PieceSelectionStrategy %d", tps.PieceSelectionStrategy)
	tps.PieceSelectionStrategy = RandomSelectionStrategy
	t.Logf("PieceSelectionStrategy %d", tps.PieceSelectionStrategy)
}

func TestMake(t *testing.T) {
	res := make([]int, 0)
	t.Logf("%v length=%d,capacity=%d", res, len(res), cap(res))
	res = make([]int, 0, 10)
	t.Logf("%v length=%d,capacity=%d", res, len(res), cap(res))
	res = make([]int, 10)
	t.Logf("%v length=%d,capacity=%d", res, len(res), cap(res))
	res = append(res, []int{1, 2, 3}...)
	t.Logf("%v length=%d,capacity=%d", res, len(res), cap(res))
	for v := range res { // index
		t.Logf("%d", v)
	}
}

func TestDomainNameResolve(t *testing.T) {
	// This function doesn't do any name resolution: both the address and the port must be numeric.
	// addrPort, err := netip.ParseAddrPort(fmt.Sprintf("%s:%d", "baidu.com", 29601))
	addrPort, err := netip.ParseAddrPort(fmt.Sprintf("%s:%d", "192.168.124.121", 29601))
	if err != nil {
		t.Errorf("Error parsing address and port: %v", err)
		return
	}
	t.Logf("%v", addrPort)
	// name resolution
	host := "baidu.com"
	port := 29601
	// Resolve the hostname to an IP address
	ips, err := net.LookupIP(host)
	if err != nil {
		t.Errorf("Error resolving hostname: %v", err)
		return
	}
	ip := ips[0]

	addrPort, err = netip.ParseAddrPort(fmt.Sprintf("%s:%d", ip, port))
	t.Logf("%v", addrPort)
}

func TestTime(t *testing.T) {
	a := time.Now()
	time.Sleep(1 * time.Second)
	b := time.Now()
	t.Logf("%v-%v=%v, %v", b, a, b.Sub(a).Microseconds(), b.Sub(a).Seconds())
	t.Logf("%v", b.Sub(a).Seconds() > 1)
}

func TestSelect(t *testing.T) {
	var wg sync.WaitGroup
	wg.Add(1)
	exit := make(chan bool)
	go func() {
		defer wg.Done()
		time.Sleep(3 * time.Second)
		exit <- true
	}()

	go func() {
		for {
			select {
			case <-exit:
				t.Logf("wg.Done")
				return
			case <-time.After(1 * time.Second):
			}
			t.Logf("time.After")
		}
	}()
	wg.Wait()
}

func TestSelectContext(t *testing.T){
	var wg sync.WaitGroup
	wg.Add(1)
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	go func() {
		defer wg.Done()
		time.Sleep(3 * time.Second)
		cancel()
	}()

	go func() {
		for {
			select {
			case <-ctx.Done():
				t.Logf("wg.Done")
				return
			case <-time.After(1 * time.Second):
			}
			t.Logf("time.After")
		}
	}()

	wg.Wait()
	time.Sleep(2 * time.Second)
}