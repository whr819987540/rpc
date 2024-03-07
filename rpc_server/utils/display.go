package utils

import (
	"context"
	"expvar"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/storage"
	"github.com/dustin/go-humanize"
)

func TorrentBar(t *torrent.Torrent, pieceStates bool, context context.Context) {
	go func() {
		start := time.Now()
		if t.Info() == nil {
			log.Printf("%v: getting torrent info for %q\n", time.Since(start), t.Name())
			<-t.GotInfo()
		}
		lastStats := t.Stats()
		var lastLine string
		interval := 3 * time.Second
		for {
			// 调用函数退出时, 终止该goroutine
			select {
			// 下载结束&打印下载信息后, 调用函数退出, 该goroutine被终止
			case <-context.Done():
				log.Printf("%v: stopping torrent bar for %q\n", time.Since(start), t.Name())
				return
			case <-time.After(interval):
			}
			var completedPieces, partialPieces int
			psrs := t.PieceStateRuns()
			for _, r := range psrs {
				if r.Complete {
					completedPieces += r.Length
				}
				if r.Partial {
					partialPieces += r.Length
				}
			}
			stats := t.Stats()
			byteRate := int64(time.Second)
			byteRate *= stats.BytesReadUsefulData.Int64() - lastStats.BytesReadUsefulData.Int64()
			byteRate /= int64(interval)
			line := fmt.Sprintf(
				"%s %v: downloading %q: %s/%s(%d Bytes written to memory), %d/%d pieces completed (%d partial): %v/s\n",
				time.Now().Format("2006-01-02 15:04:05"),
				time.Since(start),
				t.Name(),
				humanize.Bytes(uint64(t.BytesCompleted())),
				humanize.Bytes(uint64(t.Length())),
				t.BytesWrittenToMemory(),
				completedPieces,
				t.NumPieces(),
				partialPieces,
				humanize.Bytes(uint64(byteRate)),
			)
			if line != lastLine {
				lastLine = line
				os.Stdout.WriteString(line)
			}
			if pieceStates {
				log.Println(psrs)
			}
			lastStats = stats
		}
	}()
}

func WaitForPieces(ctx context.Context, t *torrent.Torrent, beginIndex, endIndex int) {
	sub := t.SubscribePieceStateChanges()
	defer sub.Close()
	expected := storage.Completion{
		Complete: true,
		Ok:       true,
	}
	pending := make(map[int]struct{})
	for i := beginIndex; i < endIndex; i++ {
		if t.Piece(i).State().Completion != expected {
			pending[i] = struct{}{}
		}
	}
	for {
		// 基于发布/订阅模式, 等待所有的piece被下载, 然后返回
		if len(pending) == 0 {
			return
		}
		select {
		// ev.Index这个piece被下载
		case ev := <-sub.Values:
			if ev.Completion == expected {
				delete(pending, ev.Index) // delete from the map by key
			}
		case <-ctx.Done():
			return
		}
	}
}

func OutputStats(cl *torrent.Client) {
	expvar.Do(func(kv expvar.KeyValue) {
		log.Printf("%s: %s\n", kv.Key, kv.Value)
	})
	cl.WriteStatus(os.Stdout)
}
