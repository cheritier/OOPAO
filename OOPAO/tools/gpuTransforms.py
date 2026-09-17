"""GPU translation matching skimage's cubic image warp for constant boundaries."""


_TRANSLATE_CUBIC = None


def translate_cubic(image, shift):
    """Translate a CuPy image by (x, y) pixels with Catmull-Rom interpolation."""
    import cupy as cp

    global _TRANSLATE_CUBIC
    if _TRANSLATE_CUBIC is None:
        _TRANSLATE_CUBIC = cp.ElementwiseKernel(
            'raw T image, int32 width, int32 height, float64 dx, float64 dy',
            'T translated',
            r'''
            const int row = i / width;
            const int col = i % width;
            const double y = row - dy;
            const double x = col - dx;
            if (y < 0.0 || y > height - 1 || x < 0.0 || x > width - 1) {
                translated = (T)0;
                return;
            }
            const int y0 = (int)floor(y);
            const int x0 = (int)floor(x);
            double value = 0.0;
            for (int ry = -1; ry <= 2; ++ry) {
                const int iy = y0 + ry;
                if (iy < 0 || iy >= height) continue;
                const double ay = fabs(y - iy);
                const double wy = ay <= 1.0
                    ? 1.5 * ay * ay * ay - 2.5 * ay * ay + 1.0
                    : (ay < 2.0 ? -0.5 * ay * ay * ay + 2.5 * ay * ay - 4.0 * ay + 2.0 : 0.0);
                for (int rx = -1; rx <= 2; ++rx) {
                    const int ix = x0 + rx;
                    if (ix < 0 || ix >= width) continue;
                    const double ax = fabs(x - ix);
                    const double wx = ax <= 1.0
                        ? 1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0
                        : (ax < 2.0 ? -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0 : 0.0);
                    value += (double)image[iy * width + ix] * wy * wx;
                }
            }
            translated = (T)value;
            ''',
            'oopao_translate_cubic')
    translated = _TRANSLATE_CUBIC(image, image.shape[1], image.shape[0],
                                  float(shift[0]), float(shift[1]),
                                  size=image.size).reshape(image.shape)
    return cp.clip(translated, cp.minimum(image.min(), 0),
                   cp.maximum(image.max(), 0))
