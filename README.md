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

	"github.com/TcMits/nsfw"
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

	d := must(nsfw.New())
	defer d.Close()

	value := must(d.Detect(context.Background(), img))
	fmt.Printf("%+v\n", value)
}
```

## Thanks

- [model](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)
- [onnx purego](https://github.com/shota3506/onnxruntime-purego)
