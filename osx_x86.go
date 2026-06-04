//go:build darwin && amd64

package nsfw

import _ "embed"

const fileName = "libonnxruntime_x64.1.23.2.dylib"

//go:embed onnxruntime/libonnxruntime_x86_64.1.23.2.dylib
var onnxRuntimeBytes []byte
