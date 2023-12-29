##/bin/bash
if [ -n "$GO" ]; then
    $GO build -o rpc_server.bin *.go 
else
    go build -o rpc_server.bin *.go 
fi