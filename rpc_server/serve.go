package main

import (
	// "bytes"
	"fmt"
	// "io/ioutil"
	// "net/http"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/bradfitz/iter"

	"github.com/anacrolix/log"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

func fromMemory(byteData []byte) (*metainfo.MetaInfo, error) {
	info := metainfo.Info{}
	mi := metainfo.MetaInfo{}
	err := info.BuildFromMemory(byteData, "from memory")
	if err != nil {
		return nil, err
	}
	mi.SetDefaults()
	mi.InfoBytes = bencode.MustMarshal(info)
	// TODO:tracker也应该是可以配置的,不应该固定
	// DONE:将tracker的值放到了配置文件中
	mi.Announce = Trackers[0][0]
	mi.AnnounceList = Trackers

	return &mi, nil
}

func fromTMPFS(filePath string) (*metainfo.MetaInfo, error) {
	// 1) get the Info which describes the filePath
	// 2) get the MetaInfo with all fields set

	// Info
	info := metainfo.Info{}
	err := info.BuildFromFilePath(filePath)
	if err != nil {
		return nil, err
	}

	// MetaInfo
	mi := metainfo.MetaInfo{}
	mi.SetDefaults()
	mi.InfoBytes = bencode.MustMarshal(info)
	// TODO:tracker也应该是可以配置的,不应该固定
	// DONE:将tracker的值放到了配置文件中
	mi.Announce = Trackers[0][0]
	mi.AnnounceList = Trackers

	return &mi, nil
}

func infoBytesToInfo(infoBytes []byte) (*metainfo.Info, error) {
	info := &metainfo.Info{}
	err := bencode.Unmarshal(infoBytes, info)
	if err != nil {
		return nil, err
	}
	return info, nil
}

func seed(mi *metainfo.MetaInfo, mbp *storage.MemoryBuf) (err error) {
	log.Printf("start seeding")
	clientConfig := torrent.NewDefaultClientConfig()
	clientConfig.Seed = true
	clientConfig.Debug = *debugFlag

	// 基于totalLength创建storage/client
	// clientConfig.DefaultStorage
	mbpp := &mbp
	storageImplCloser, err := storage.NewMemory(mbp.Length, mbpp)
	// storageImplCloser := storage.NewFile("./")
	if err != nil {
		return fmt.Errorf("NewMemory storage %w", err)
	}
	clientConfig.DefaultStorage = storageImplCloser

	client, err := torrent.NewClient(clientConfig)
	if err != nil {
		return fmt.Errorf("NewClient %w", err)
	}

	// 向client中添加torrent
	_, err = client.AddTorrent(mi)
	if err != nil {
		return fmt.Errorf("AddTorrent: %w", err)
	}
	return
}

func seedFromTMPFS(mip *metainfo.MetaInfo) error {
	// 1) create a client
	// 2) add the MetaInfo to the client and return a torrent
	// 3) when MetaInfo added, seeding starts

	// // client config
	// clientConfig := torrent.NewDefaultClientConfig()
	// clientConfig.Seed = true
	// clientConfig.Debug = *debugFlag
	// // 指定torrent data的存储路径
	// storageImplCloser := storage.NewFile(configStruct.Model.ModelPath)
	// clientConfig.DefaultStorage = storageImplCloser

	// // client
	// cl, err := torrent.NewClient(clientConfig)
	// if err != nil {
	// 	return fmt.Errorf("new torrent client: %w", err)
	// }

	// add torrent
	t, err := torrentClient.AddTorrent(mip)
	if err != nil {
		log.Printf("add torrent error: %v", err)
		return err
	}

	// print the MetaInfo
	mi := t.Metainfo()
	log.Printf("Metainfo in torrent")
	pprintMetainfo(&mi, pprintMetainfoFlags{
		JustName:    false,
		PieceHashes: false,
		Files:       false,
	})

	info, err := mi.UnmarshalInfo()
	log.Printf("info: %v", info.Describe())

	_, err = os.Stat("./torrent")
	if err != nil {
		if os.IsNotExist(err) {
			err = os.MkdirAll("./torrent", 0755)
			if err != nil {
				log.Printf("./torrent not exist, but failed to create it.")
			} else {
				log.Printf("./torrent not exist, created.")
			}
		} else {
			log.Print(err)
			return err
		}
	}

	path := fmt.Sprintf("./torrent/%s.torrent", info.BestName())
	err = writeMetainfoToFile(mi, path)
	if err != nil {
		log.Printf("error writing %q: %v", path, err)
		return err
	} else {
		log.Printf("wrote %q", path)
	}

	return nil

	// // 1) get the Info which describes the filePath
	// // 2) create a client
	// // 3) add the Info to the client and return a torrent
	// // 4) fullfill the tracker field of the torrent

	// // Info
	// info := metainfo.Info{}
	// err = info.BuildFromFilePath(filePath)
	// if err != nil {
	// 	return err
	// } else {
	// 	log.Printf("build info from TMPFS file path: %v", info.Describe())
	// }

	// // MetaInfo, for getting Hash(Bytes(Info)), no other usage
	// mi := metainfo.MetaInfo{
	// 	InfoBytes: bencode.MustMarshal(info),
	// }
	// log.Printf("build info from TMPFS file path: %v", mi.Describe())
	// ih := mi.HashInfoBytes()

	// // client config
	// log.Printf("start seeding")
	// clientConfig := torrent.NewDefaultClientConfig()
	// clientConfig.Seed = true
	// clientConfig.Debug = *debugFlag

	// // client
	// cl, err := torrent.NewClient(clientConfig)
	// if err != nil {
	// 	return fmt.Errorf("new torrent client: %w", err)
	// }
	// defer cl.Close()

	// // piece completion storage
	// pc, err := storage.NewDefaultPieceCompletionForDir(".")
	// if err != nil {
	// 	return fmt.Errorf("new piece completion: %w", err)
	// }
	// defer pc.Close()

	// // designate the dst path to be saved and the piece completion storage
	// to, _ := cl.AddTorrentOpt(torrent.AddTorrentOpts{
	// 	// identify a torrent
	// 	InfoHash: ih,
	// 	Storage: storage.NewFileOpts(storage.NewFileClientOpts{
	// 		// 该torrent指定文件或目录的路径
	// 		// 对于seeder, 是已存在文件的路径
	// 		ClientBaseDir: filePath,
	// 		FilePathMaker: func(opts storage.FilePathMakerOpts) string {
	// 			return filepath.Join(opts.File.Path...)
	// 		},
	// 		TorrentDirMaker: nil,
	// 		PieceCompletion: pc,
	// 	}),
	// 	InfoBytes: mi.InfoBytes,
	// })
	// defer to.Drop() // drop this torrent from the client torrent list after seeding

	// // set the tracker field
	// err = to.MergeSpec(&torrent.TorrentSpec{
	// 	Trackers: Trackers,
	// },
	// )
	// if err != nil {
	// 	return fmt.Errorf("setting trackers: %w", err)
	// } else {
	// 	log.Print("set the tracker field ok")
	// }

	// // print the MetaInfo
	// mi = to.Metainfo()
	// pprintMetainfo(&mi, pprintMetainfoFlags{
	// 	JustName:    false,
	// 	PieceHashes: false,
	// 	Files:       false,
	// })

	// // print the Info
	// info, err = mi.UnmarshalInfo()
	// log.Printf("info: %v",info.Describe())

	// path := fmt.Sprintf("%s.torrent", info.BestName())
	// err = writeMetainfoToFile(mi, path)
	// if err == nil {
	// 	log.Printf("wrote %q", path)
	// } else {
	// 	log.Printf("error writing %q: %v", path, err)
	// }
	// select {}
}

func writeMetainfoToFile(mi metainfo.MetaInfo, path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o640)
	if err != nil {
		return err
	}
	defer f.Close()
	err = mi.Write(f)
	if err != nil {
		return err
	}
	return f.Close()
}

func seed_another(byteData []byte, filePath string) error {
	log.Printf("serve")
	cfg := torrent.NewDefaultClientConfig()
	cfg.Seed = true
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		return fmt.Errorf("new torrent client: %w", err)
	}
	defer cl.Close()

	info := metainfo.Info{}
	// err = info.BuildFromFilePath(filePath)
	err = info.BuildFromMemory(byteData, "from memory")
	log.Print(info.Pieces[:20])

	if err != nil {
		return fmt.Errorf("building info from memory: %w", err)
	}
	for _, fi := range info.Files {
		log.Printf("added %q", fi.Path)
	}

	mi := metainfo.MetaInfo{
		InfoBytes: bencode.MustMarshal(info),
	}

	pc, err := storage.NewDefaultPieceCompletionForDir(".")
	if err != nil {
		return fmt.Errorf("new piece completion: %w", err)
	}
	defer pc.Close()

	ih := mi.HashInfoBytes()
	to, _ := cl.AddTorrentOpt(torrent.AddTorrentOpts{
		InfoHash: ih,
		Storage: storage.NewFileOpts(storage.NewFileClientOpts{
			ClientBaseDir: filePath,
			FilePathMaker: func(opts storage.FilePathMakerOpts) string {
				return filepath.Join(opts.File.Path...)
			},
			TorrentDirMaker: nil,
			PieceCompletion: pc,
		}),
		InfoBytes: mi.InfoBytes,
	})
	defer to.Drop() // drop this torrent from the client torrent list after seeding

	err = to.MergeSpec(&torrent.TorrentSpec{
		Trackers: [][]string{
			{`wss://tracker.btorrent.xyz`},
			{`wss://tracker.openwebtorrent.com`},
			{"http://p4p.arenabg.com:1337/announce"},
			{"udp://tracker.opentrackr.org:1337/announce"},
			{"udp://tracker.openbittorrent.com:6969/announce"},
			{"udp://tracker.moeking.me:6969/announce"},
			{"udp://p4p.arenabg.com:1337/announce"},
		},
	})
	if err != nil {
		return fmt.Errorf("setting trackers: %w", err)
	}
	mi = to.Metainfo()
	// log.Print(mi.InfoBytes)
	pprintMetainfo(&mi, pprintMetainfoFlags{
		JustName:    false,
		PieceHashes: false,
		Files:       false})

	fmt.Printf("%v: %v\n", to, to.Metainfo().Magnet(&ih, &info))

	path := fmt.Sprintf("%s.torrent", info.BestName())
	mi.Write(os.Stdout)
	if err == nil {
		log.Printf("wrote %q", path)
	} else {
		log.Printf("error writing %q: %v", path, err)
	}

	select {}

}

type pprintMetainfoFlags struct {
	JustName    bool
	PieceHashes bool
	Files       bool
}

func pprintMetainfo(metainfo *metainfo.MetaInfo, flags pprintMetainfoFlags) error {
	info, err := metainfo.UnmarshalInfo()
	if err != nil {
		return fmt.Errorf("error unmarshalling info: %s", err)
	}
	if flags.JustName {
		fmt.Printf("%s\n", info.Name)
		return nil
	}
	d := map[string]interface{}{
		"Name":         info.Name,
		"Name.Utf8":    info.NameUtf8,
		"NumPieces":    info.NumPieces(),
		"PieceLength":  info.PieceLength,
		"InfoHash":     metainfo.HashInfoBytes().HexString(),
		"NumFiles":     len(info.UpvertedFiles()),
		"TotalLength":  info.TotalLength(),
		"Announce":     metainfo.Announce,
		"AnnounceList": metainfo.AnnounceList,
		"UrlList":      metainfo.UrlList,
	}
	if len(metainfo.Nodes) > 0 {
		d["Nodes"] = metainfo.Nodes
	}
	if flags.Files {
		d["Files"] = info.UpvertedFiles()
	}
	if flags.PieceHashes {
		d["PieceHashes"] = func() (ret []string) {
			for i := range iter.N(info.NumPieces()) {
				ret = append(ret, hex.EncodeToString(info.Pieces[i*20:(i+1)*20]))
			}
			return
		}()
	}
	b, _ := json.MarshalIndent(d, "", "  ")
	_, err = os.Stdout.Write(b)
	return err
}
