package bundle

import (
	"context"
	"errors"
	"image"
	"os"
	"path"

	"github.com/TcMits/nsfw"
	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

var ortPath string

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

func GetRuntime() (*ort.Runtime, error) {
	return ort.NewRuntime(ortPath, 23)
}

type Detector struct {
	runtime *ort.Runtime
	env     *ort.Env
	session *nsfw.DetectSession
}

func New(opts *ort.SessionOptions) (*Detector, error) {
	rt, err := GetRuntime()
	if err != nil {
		return nil, err
	}

	env, err := rt.NewEnv("nsfw", ort.LoggingLevelWarning)
	if err != nil {
		return nil, err
	}

	ss, err := nsfw.New(rt, env, opts)
	if err != nil {
		return nil, err
	}

	return &Detector{
		runtime: rt,
		env:     env,
		session: ss,
	}, nil
}

func (d *Detector) Close() error {
	errs := make([]error, 0, 2)
	errs = append(errs, d.session.Close())
	d.env.Close()
	errs = append(errs, d.runtime.Close())
	return errors.Join(errs...)
}

func (d *Detector) Detect(ctx context.Context, img image.Image) (nsfw.Labels, error) {
	return d.session.Detect(ctx, img)
}
