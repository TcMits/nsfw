//go:build darwin && arm64

package nsfw

import _ "embed"

const fileName = "libonnxruntime_arm64.1.23.2.dylib"

//go:embed onnxruntime/libonnxruntime_arm64.1.23.2.dylib
var onnxRuntimeBytes []byte
