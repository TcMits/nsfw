//go:build linux && amd64

package nsfw

import _ "embed"

const fileName = "libonnxruntime_x64.so.1.23.2"

//go:embed onnxruntime/libonnxruntime_x64.so.1.23.2
var onnxRuntimeBytes []byte
