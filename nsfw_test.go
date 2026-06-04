package nsfw

import (
	"bytes"
	"context"
	"image"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"
)

func Test_MultipleRuntime(t *testing.T) {
	rt1 := must(New())
	defer rt1.Close()
	rt2 := must(New())
	defer rt2.Close()
	rt3 := must(New())
	defer rt3.Close()
}

func TestIsSafe(t *testing.T) {
	detector := must(New())
	defer detector.Close()

	detect := func(img []byte) (bool, Labels) {
		imgs, _, err := image.Decode(bytes.NewReader(img))
		must1(err)
		labels := must(detector.Detect(context.Background(), imgs))
		return labels.Normal > 0.63, labels
	}

	imageFiles := os.DirFS("./testdata")
	if err := fs.WalkDir(imageFiles, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			panic(err)
		}

		if d.IsDir() || strings.HasPrefix(filepath.Base(path), ".") {
			return nil
		}

		t.Run(path, func(t *testing.T) {
			file := must(imageFiles.Open(path))
			defer file.Close()

			buf := must(io.ReadAll(file))

			safe, l := detect(buf)

			basename := filepath.Base(path)

			t.Logf("labels:   %+v", l)
			isSafe := !(strings.Contains(basename, "porn") ||
				strings.Contains(basename, "hentai") ||
				strings.Contains(basename, "sexy"))
			if isSafe != safe {
				t.Errorf("expected safe image")
			}
		})

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkDetect(b *testing.B) {
	imageFiles := os.DirFS("./testdata")
	file := must(imageFiles.Open("brooke-cagle-9fHMo1-5Io8-unsplash.jpg"))
	defer file.Close()
	img, _, err := (image.Decode(file))
	must1(err)

	detector := must(New())
	defer detector.Close()

	for b.Loop() {
		detector.Detect(context.Background(), img)
	}
}

func BenchmarkParallelDetect(b *testing.B) {
	imageFiles := os.DirFS("./testdata")
	file := must(imageFiles.Open("brooke-cagle-9fHMo1-5Io8-unsplash.jpg"))
	defer file.Close()
	img, _, err := (image.Decode(file))
	must1(err)

	detector := must(New())
	defer detector.Close()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			detector.Detect(context.Background(), img)
		}
	})
}
