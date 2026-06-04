package nsfw

import (
	"bytes"
	"context"
	_ "embed"
	"image"
	"math"
	"os"
	"path"
	"sync"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
	"golang.org/x/image/draw"
)

//go:embed model.onnx
var onnxModelBytes []byte

const (
	width   = 384
	height  = 384
	channel = 3
	mean    = 0.5
	std     = 0.5
	scale   = 255.0
)

var (
	ortPath string
	rect    = image.Rect(0, 0, width, height)
	pool    = sync.Pool{New: func() any {
		return &poolEntry{
			inputTensor:  make([]float32, channel*height*width),
			resizedImage: image.NewNRGBA(rect),
		}
	}}
)

func init() {
	ortPath = path.Join(os.TempDir(), "onnxruntime/"+fileName)

	if _, err := os.Stat(ortPath); err != nil {
		if !os.IsNotExist(err) {
			panic(err)
		}

		must1(os.MkdirAll(path.Dir(ortPath), os.ModePerm))
		must1(os.WriteFile(ortPath, onnxRuntimeBytes, os.ModePerm))
	}
}

func must1(err error) {
	if err != nil {
		panic(err)
	}
}

func must[T any](v T, err error) T {
	must1(err)
	return v
}

type Detector struct {
	runtime *ort.Runtime
	env     *ort.Env
	session *ort.Session
}

func New() (*Detector, error) {
	rt, err := ort.NewRuntime(ortPath, 23)
	if err != nil {
		return nil, err
	}

	env, err := rt.NewEnv("nsfw", ort.LoggingLevelWarning)
	if err != nil {
		return nil, err
	}

	sess, err := rt.NewSessionFromReader(env, bytes.NewReader(onnxModelBytes), nil)
	if err != nil {
		return nil, err
	}

	return &Detector{
		runtime: rt,
		env:     env,
		session: sess,
	}, nil
}

func (d *Detector) Close() error {
	d.session.Close()
	d.env.Close()
	return d.runtime.Close()
}

type poolEntry struct {
	inputTensor  []float32
	resizedImage *image.NRGBA
}

type Labels struct {
	Normal float32 `json:"normal"`
	NSFW   float32 `json:"nsfw"`
}

func (d *Detector) Detect(ctx context.Context, img image.Image) (Labels, error) {
	entry := pool.Get().(*poolEntry)
	defer pool.Put(entry)

	draw.BiLinear.Scale(entry.resizedImage, rect, img, img.Bounds(), draw.Over, nil)
	for y := range height {
		for x := range width {
			col := entry.resizedImage.At(x, y)
			r, g, b, _ := col.RGBA()
			entry.inputTensor[0*height*width+y*width+x] = float32(((float64(r)/255.0)/scale - mean) / std)
			entry.inputTensor[1*height*width+y*width+x] = float32(((float64(g)/255.0)/scale - mean) / std)
			entry.inputTensor[2*height*width+y*width+x] = float32(((float64(b)/255.0)/scale - mean) / std)
		}
	}

	tensor, err := ort.NewTensorValue(d.runtime, entry.inputTensor, []int64{1, channel, height, width})
	if err != nil {
		return Labels{}, err
	}
	defer tensor.Close()

	result, err := d.session.Run(ctx, map[string]*ort.Value{"pixel_values": tensor}, ort.WithOutputNames("logits"))
	if err != nil {
		return Labels{}, err
	}

	logits, _, err := ort.GetTensorData[float32](result["logits"])
	if err != nil {
		return Labels{}, err
	}

	softMax(logits)
	return Labels{Normal: logits[0], NSFW: logits[1]}, nil
}

func softMax(input []float32) {
	if len(input) == 0 {
		return
	}

	s := 0.0
	c := input[0]
	for _, e := range input[1:] {
		if e > c {
			c = e
		}
	}

	for _, e := range input {
		s += math.Exp(float64(e - c))
	}

	if s == 0 {
		return
	}

	for i, v := range input {
		input[i] = float32(math.Exp(float64(v-c)) / s)
	}
}
