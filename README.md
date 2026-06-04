# NSFW

Simple nsfw image detection in go (CPU + OSX/Linux)

## Examples

```go
package main

import (
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	"os"

	"github.com/TcMits/nsfw/bundle"
)

func must1(err error) {
	if err != nil {
		panic(err)
	}
}

func must[T any](v T, err error) T {
	must1(err)
	return v
}

func main() {
	file := must(os.Open("testdata/sexy_1.jpg"))
	defer file.Close()

	img, _, err := image.Decode(file)
	must1(err)

	d := must(bundle.New(nil))
	defer d.Close()

	value := must(d.Detect(context.Background(), img))
	fmt.Printf("%+v\n", value)
}
```

## Benchmark

```sh
➜  nsfw git:(main) ✗ go test -bench=. -benchmem -run=^$ -benchtime 30s
goos: darwin
goarch: arm64
pkg: github.com/TcMits/nsfw
cpu: Apple M4 Pro
BenchmarkDetect-14                        	     176	 203457782 ns/op	67587530 B/op	      76 allocs/op
BenchmarkDetectORTOneThread-14            	      67	 525527306 ns/op	67609384 B/op	      76 allocs/op
BenchmarkParallelDetect-14                	     566	  61511094 ns/op	68225821 B/op	      76 allocs/op
BenchmarkParallelDetectORTOneThread-14    	     560	  54582056 ns/op	68232883 B/op	      76 allocs/op
```

## Thanks

- [model](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)
- [onnx purego](https://github.com/shota3506/onnxruntime-purego)
