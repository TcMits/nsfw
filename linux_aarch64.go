//go:build linux && arm64

package nsfw

import _ "embed"

const fileName = "libonnxruntime_aarch64.so.1.23.2"

//go:embed onnxruntime/libonnxruntime_aarch64.so.1.23.2
var onnxRuntimeBytes []byte
