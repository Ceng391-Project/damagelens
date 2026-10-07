from collections.abc import Iterator


def tiles(height: int, width: int, size: int = 512, stride: int | None = None) -> Iterator[tuple[int, int]]:
    # the last row/column is pulled back inside the image instead of padding
    stride = stride or size
    ys = list(range(0, max(height - size, 0) + 1, stride))
    xs = list(range(0, max(width - size, 0) + 1, stride))
    if ys[-1] + size < height:
        ys.append(height - size)
    if xs[-1] + size < width:
        xs.append(width - size)
    for y in ys:
        for x in xs:
            yield y, x
